import os
import json
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import torch
import uuid
import tempfile
from typing import Dict
from pathlib import Path
from helpers import cleanup
import time
import psutil
from audio_quality import analyze_media_quality

app = FastAPI()

# Dictionary to store job statuses
job_statuses = {}


class TranscriptionProgress:
    def __init__(self, job_id: str, temp_dir: Path):
        self.job_id = job_id
        self.temp_dir = temp_dir
        self.progress_file = temp_dir / f"{job_id}_progress.json"
        self.create_progress_file()

    def create_progress_file(self):
        with open(self.progress_file, "w") as f:
            json.dump(
                {"status": "processing", "stage": "initializing", "progress": 0}, f
            )

    def mark_complete(self, output_file: str):
        with open(self.progress_file, "w") as f:
            json.dump(
                {"status": "completed", "output_file": output_file, "progress": 100}, f
            )
        # Update the job_statuses dictionary
        job_statuses[self.job_id] = {"status": "completed", "output_file": output_file}

    def get_status(self):
        try:
            if self.progress_file.exists():
                with open(self.progress_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return {"status": "processing", "stage": "initializing", "progress": 0}


def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Terminate parent
        parent.terminate()

        # Wait for processes to terminate
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)

        # Force kill if still alive
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


# Create a base directory for all temporary files
BASE_TEMP_DIR = Path("temp_processing")
BASE_TEMP_DIR.mkdir(exist_ok=True)


def get_job_dir(job_id: str) -> Path:
    """Create and return a unique directory for each job"""
    job_dir = BASE_TEMP_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    return job_dir


def cleanup_job_files(job_id: str):
    """Clean up all files associated with a job"""
    job_dir = BASE_TEMP_DIR / job_id
    if job_dir.exists():
        import shutil

        shutil.rmtree(job_dir)


def process_audio(
    job_id: str,
    audio_path: str,
    language: str,
    model_name: str,
    device: str,
    suppress_numerals: bool,
    stem: bool,
):
    progress = job_statuses[job_id]
    try:
        job_dir = get_job_dir(job_id)
        output_base = os.path.splitext(audio_path)[0]

        command = [
            "python",
            "transcribe.py",
            "-a",
            audio_path,
            "--whisper-model",
            model_name,
            "--device",
            device,
            "--temp-dir",
            str(job_dir),
            "--job-id",
            job_id,
        ]
        if language:
            command.extend(["--language", language])
        if suppress_numerals:
            command.append("--suppress_numerals")
        if not stem:
            command.append("--no-stem")

        process = subprocess.Popen(command)

        while process.poll() is None:
            status = job_statuses[job_id]
            if isinstance(status, dict) and status.get("canceled", False):
                terminate_process_tree(process.pid)
                cleanup(str(job_dir))
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return
            time.sleep(0.5)

        output_file = f"{output_base}.txt"
        if os.path.exists(output_file):
            final_output = job_dir / "results" / "transcription.txt"
            results_dir = job_dir / "results"
            results_dir.mkdir(exist_ok=True)

            import shutil

            shutil.copy2(output_file, final_output)
            os.remove(output_file)

            if isinstance(progress, TranscriptionProgress):
                progress.mark_complete(str(final_output))
            job_statuses[job_id] = {
                "status": "completed",
                "output_file": str(final_output),
            }
        else:
            if not (isinstance(progress, dict) and progress.get("canceled")):
                raise FileNotFoundError("Transcription output file not found")

    except Exception as e:
        if not job_statuses[job_id].get("canceled"):
            job_statuses[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if process and process.poll() is None:
            terminate_process_tree(process.pid)
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    language: str = Form(None),
    model_name: str = Form("large-v3"),
    device: str = Form(None),
    suppress_numerals: bool = Form(False),
    stem: bool = Form(True),
):
    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)

    # Save the uploaded file
    audio_path = job_dir / f"input_{job_id}.wav"
    with open(audio_path, "wb") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create progress tracker with job directory
    job_statuses[job_id] = TranscriptionProgress(job_id, job_dir)

    background_tasks.add_task(
        process_audio,
        job_id,
        str(audio_path),
        language,
        model_name,
        device,
        suppress_numerals,
        stem,
    )

    return JSONResponse({"job_id": job_id, "status": "processing"})


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    status = job_statuses[job_id]
    if isinstance(status, TranscriptionProgress):
        # Read progress file directly
        progress_file = Path(status.progress_file)
        try:
            if progress_file.exists():
                with open(progress_file) as f:
                    progress_data = json.load(f)
                    return JSONResponse(
                        {
                            "status": "processing",
                            "stage": progress_data.get("stage", "initializing"),
                            "progress": progress_data.get("progress", 0),
                        }
                    )
        except Exception:
            pass
        return JSONResponse({"status": "processing", "stage": "initializing"})

    # Handle completed/failed/canceled states
    return JSONResponse(status)


@app.get("/result/{job_id}")
async def get_result(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    job_status = job_statuses[job_id]

    # Check if it's still a TranscriptionProgress object
    if isinstance(job_status, TranscriptionProgress):
        # Read the progress file to get the actual status
        status = job_status.get_status()
        if status.get("status") != "completed":
            return JSONResponse({"status": "processing"}, status_code=400)
        # If completed, use the status from the progress file
        job_status = status

    # Handle dictionary status
    if isinstance(job_status, dict):
        if job_status.get("status") != "completed":
            return JSONResponse(
                {"status": job_status.get("status", "unknown")}, status_code=400
            )

        output_file = job_status.get("output_file")
        if not output_file or not os.path.exists(output_file):
            return JSONResponse({"status": "file_not_found"}, status_code=404)

        # Add cleanup task
        background_tasks.add_task(cleanup_job_files, job_id)

        # Return the file
        return FileResponse(
            output_file, media_type="text/plain", filename="transcription.txt"
        )

    return JSONResponse({"status": "invalid_state"}, status_code=500)


@app.post("/cancel/{job_id}")
async def cancel_transcription(job_id: str):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    job_statuses[job_id] = {"status": "canceled", "canceled": True, "progress": 0}

    return JSONResponse({"status": "canceled"})


@app.post("/calculate-snr")
async def calculate_snr(audio: UploadFile = File(...)) -> Dict[str, float]:
    """
    Calculate SNR from uploaded audio file using existing analyzer
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(audio.filename)[1]
    ) as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Use your existing analyze_media_quality function
        return analyze_media_quality(temp_path)
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
