import os
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import torch
import uuid
from pathlib import Path
from helpers import cleanup

app = FastAPI()

# Dictionary to store job statuses
job_statuses = {}

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
        ]
        if language:
            command.extend(["--language", language])
        if suppress_numerals:
            command.append("--suppress_numerals")
        if not stem:
            command.append("--no-stem")

        # Start the transcription as a subprocess
        process = subprocess.Popen(command)

        # Periodically check if the job was canceled
        while process.poll() is None:
            if job_statuses[job_id].get("canceled"):
                process.terminate()  # Stop the subprocess if canceled
                job_statuses[job_id] = {"status": "canceled"}  # Update status
                cleanup(str(job_dir))  # Call your cleanup function
                return  # Exit the function to skip further processing
        process.wait()  # Ensure the process finishes

        # If not canceled, proceed with file handling
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
            # Handle missing file without error if canceled
            if not job_statuses[job_id].get("canceled"):
                raise FileNotFoundError("Transcription output file not found")

    except Exception as e:
        job_statuses[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # Clean up the audio file after processing
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

    # Set initial job status
    job_statuses[job_id] = {"status": "processing"}

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
    return JSONResponse(job_statuses[job_id])


@app.get("/result/{job_id}")
async def get_result(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    job_status = job_statuses[job_id]
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
    job_statuses[job_id]["canceled"] = True
    return JSONResponse({"status": "canceled"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
