import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import torch
import uuid

app = FastAPI()

# Dictionary to store job statuses
job_statuses = {}


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
        # Run the diarization script
        output_base = os.path.splitext(audio_path)[0]
        command = [
            "python",
            "diarize_parallel.py",
            "-a",
            audio_path,
            "--whisper-model",
            model_name,
            "--device",
            device,
        ]
        if language:
            command.extend(["--language", language])
        if suppress_numerals:
            command.append("--suppress_numerals")
        if not stem:
            command.append("--no-stem")

        subprocess.run(command, check=True)

        # Update job status
        job_statuses[job_id] = {
            "status": "completed",
            "output_file": f"{output_base}.txt",
        }
    except Exception as e:
        job_statuses[job_id] = {"status": "failed", "error": str(e)}


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

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    # Set default device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set initial job status
    job_statuses[job_id] = {"status": "processing"}

    # Start the background task
    background_tasks.add_task(
        process_audio,
        job_id,
        temp_audio_path,
        language,
        model_name,
        device,
        suppress_numerals,
        stem,
    )

    # Return the job ID immediately
    return JSONResponse({"job_id": job_id, "status": "processing"})


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(job_statuses[job_id])


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in job_statuses:
        return JSONResponse({"status": "not_found"}, status_code=404)

    job_status = job_statuses[job_id]
    if job_status["status"] != "completed":
        return JSONResponse({"status": job_status["status"]}, status_code=400)

    output_file = job_status["output_file"]
    if not os.path.exists(output_file):
        return JSONResponse({"status": "file_not_found"}, status_code=404)

    return FileResponse(
        output_file, media_type="text/plain", filename="transcription.txt"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
