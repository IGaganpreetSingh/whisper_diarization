import os
import json
import time
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import torch
import uuid
from pathlib import Path
import sys
from helpers import cleanup

app = FastAPI()

# Dictionary to store job statuses
job_statuses = {}


class TranscriptionProgress:
    def __init__(self):
        self.current_stage = "initializing"

    def update_stage(self, stage: str):
        """Update the current processing stage"""
        self.current_stage = stage

    def get_status(self):
        return {"status": "processing", "stage": self.current_stage}

    def get(self, key, default=None):
        if key == "canceled":
            return default
        if key == "status":
            return "processing"
        if key == "stage":
            return self.current_stage
        return default


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
    try:
        job_dir = get_job_dir(job_id)
        output_base = os.path.splitext(audio_path)[0]

        progress_tracker = TranscriptionProgress()
        job_statuses[job_id] = progress_tracker

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

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)

            if process.poll() is not None:
                for output in process.stdout.readlines():
                    if output:
                        print(output.strip())
                for error in process.stderr.readlines():
                    if error:
                        print(error.strip(), file=sys.stderr)
                break

            progress_file = os.path.join(job_dir, f"{job_id}_progress.json")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, "r") as f:
                        progress_data = json.load(f)
                        progress_tracker.update_stage(progress_data["stage"])
                except Exception as e:
                    print(f"Failed to read progress: {str(e)}")

            if isinstance(job_statuses[job_id], dict) and job_statuses[job_id].get(
                "canceled"
            ):
                process.terminate()
                job_statuses[job_id] = {"status": "canceled"}
                cleanup(str(job_dir))
                return

            time.sleep(0.1)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        output_file = f"{output_base}.txt"
        if os.path.exists(output_file):
            import shutil

            final_output = job_dir / "results" / "transcription.txt"
            results_dir = job_dir / "results"
            results_dir.mkdir(exist_ok=True)
            shutil.copy2(output_file, final_output)
            os.remove(output_file)
            job_statuses[job_id] = {
                "status": "completed",
                "output_file": str(final_output),
            }
        else:
            if not (
                isinstance(job_statuses[job_id], dict)
                and job_statuses[job_id].get("canceled")
            ):
                raise FileNotFoundError("Transcription output file not found")

    except Exception as e:
        job_statuses[job_id] = {"status": "failed", "error": str(e)}
        print(f"Error in process_audio: {str(e)}", file=sys.stderr)
    finally:
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
    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Create job directory
    job_dir = get_job_dir(job_id)

    # Save the uploaded file to the job directory
    audio_path = job_dir / f"input_{job_id}.wav"
    with open(audio_path, "wb") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)

    # Set default device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create progress tracker
    job_statuses[job_id] = TranscriptionProgress()

    # Start the background task
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
        return JSONResponse(status.get_status())
    return JSONResponse(status)


@app.get("/result/{job_id}")
async def get_result(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    job_status = job_statuses[job_id]
    if isinstance(job_status, TranscriptionProgress):
        return JSONResponse({"status": "processing"}, status_code=400)

    if job_status["status"] != "completed":
        return JSONResponse({"status": job_status["status"]}, status_code=400)

    output_file = job_status["output_file"]
    if not os.path.exists(output_file):
        return JSONResponse({"status": "file_not_found"}, status_code=404)

    # Add cleanup task properly
    background_tasks.add_task(cleanup_job_files, job_id)

    # Return the file
    return FileResponse(
        output_file, media_type="text/plain", filename="transcription.txt"
    )


@app.post("/cancel/{job_id}")
async def cancel_transcription(job_id: str):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    # Handle both progress tracker and dictionary cases
    if isinstance(job_statuses[job_id], TranscriptionProgress):
        job_statuses[job_id] = {"status": "canceled", "progress": 0}
    else:
        job_statuses[job_id]["canceled"] = True

    return JSONResponse({"status": "canceled"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
