import argparse
import os
import re
import io
import faster_whisper
import torch
import torchaudio
import librosa
import soundfile as sf
import json
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from audio_quality import analyze_media_quality
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import warnings
from transformers import logging
from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    format_transcript,
    write_srt,
)
import numpy as np

logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore", message="You seem to be using the pipelines sequentially on GPU.*"
)

def update_progress(job_id, stage):
    """Update progress through file system"""
    if not job_id:
        return

    # Define stage percentages
    STAGES = {
        "initializing": 0,
        "source_separation_started": 10,
        "source_separation_loading": 15,
        "source_separation_processing": 20,
        "source_separation_completed": 30,
        "transcription_started": 35,
        "transcription_completed": 50,
        "alignment_started": 55,
        "alignment_completed": 70,
        "diarization_started": 75,
        "diarization_model_loading": 80,
        "diarization_processing": 85,
        "diarization_completed": 90,
        "finalizing_started": 92,
        "punctuation_restoration": 95,
        "generating_transcript": 97,
        "saving_files": 98,
        "completed": 100,
    }

    progress_file = os.path.join(args.temp_dir, f"{job_id}_progress.json")
    try:
        with open(progress_file, "w") as f:
            json.dump(
                {
                    "status": "processing",
                    "stage": stage,
                    "progress": STAGES.get(stage, 0),
                },
                f,
            )
    except Exception as e:
        logging.warning(f"Failed to update progress: {str(e)}")


def enhance_vocals(audio_path, output_path, volume_boost_db=6.0):
    """Enhance the isolated vocals"""
    audio, sr = librosa.load(audio_path, sr=None)
    audio_norm = librosa.util.normalize(audio)
    boost_factor = np.power(10.0, volume_boost_db / 20.0)
    audio_boosted = audio_norm * boost_factor

    def compress(signal, threshold=0.3, ratio=2.0):
        compressed = np.zeros_like(signal)
        for i, sample in enumerate(signal):
            if abs(sample) > threshold:
                if sample > 0:
                    compressed[i] = threshold + (sample - threshold) / ratio
                else:
                    compressed[i] = -(threshold + (abs(sample) - threshold) / ratio)
            else:
                compressed[i] = sample
        return compressed

    audio_compressed = compress(audio_boosted)
    audio_final = librosa.util.normalize(audio_compressed)
    sf.write(output_path, audio_final, sr)
    return output_path


mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large-v3",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument(
    "--temp-dir", type=str, help="Directory for temporary files", required=True
)
parser.add_argument("--job-id", type=str, help="Job ID for progress tracking")

args = parser.parse_args()

# Create temp directories
temp_outputs = os.path.join(args.temp_dir, "temp_outputs")
os.makedirs(temp_outputs, exist_ok=True)

# Create NeMo specific directories within temp_outputs
nemo_temp_path = os.path.join(temp_outputs, "nemo")
os.makedirs(nemo_temp_path, exist_ok=True)

language = process_language_arg(args.language, args.model_name)
update_progress(args.job_id, "initializing")
print("Initializing")

if args.stemming:
    try:
        quality_metrics = analyze_media_quality(args.audio)
        print("\nAudio Quality Analysis:")
        for metric, value in quality_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
        update_progress(args.job_id, "source_separation_loading")
        print("Source separation loading")
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "{temp_outputs}"'
        )

        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. "
                "Use --no-stem argument to disable it."
            )
            vocal_target = args.audio
        else:
            update_progress(args.job_id, "source_separation_processing")
            print("Source separation processing")
            if quality_metrics.get("snr_db", 0.0) >= 20.0: # If SNR is above threshold, use the original vocals
                vocal_target = os.path.join(
                temp_outputs,
                "htdemucs",
                os.path.splitext(os.path.basename(args.audio))[0],
                    "vocals.wav",
                )
                update_progress(args.job_id, "source_separation_completed")
                print("Source separation completed")
            else: # If SNR is below threshold, enhance the vocals
                vocals_path = os.path.join(
                    temp_outputs,
                    "htdemucs",
                    os.path.splitext(os.path.basename(args.audio))[0],
                    "vocals.wav",
                )

                update_progress(args.job_id, "source_separation_enhancing")
                print("Source separation enhancing")
                # Add vocal enhancement step
                enhanced_vocals_path = os.path.join(
                    temp_outputs,
                    "htdemucs",
                    os.path.splitext(os.path.basename(args.audio))[0],
                    "vocals_enhanced.wav",
                )
                vocal_target = enhance_vocals(
                    vocals_path, enhanced_vocals_path, volume_boost_db=6.0
                )

            update_progress(args.job_id, "source_separation_completed")
            print("Source separation completed")
            quality_metrics = analyze_media_quality(vocal_target)
            print("\nAudio Quality Analysis after source separation:")
            for metric, value in quality_metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value}")
    except Exception as e:
        logging.error(f"Error in source separation: {str(e)}")
        vocal_target = args.audio
else:
    vocal_target = args.audio

# Transcription stage
update_progress(args.job_id, "transcription_started")
print("Transcription started")

whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device]
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)

if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        batch_size=args.batch_size,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        vad_filter=True,
    )
full_transcript = "".join(segment.text for segment in transcript_segments)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

update_progress(args.job_id, "transcription_completed")
print("Transcription completed")

# Alignment stage
update_progress(args.job_id, "alignment_started")
print("Alignment started")

alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)
update_progress(args.job_id, "alignment_completed")
print("Alignment completed")
# Diarization stage
update_progress(args.job_id, "diarization_started")
print("Diarization started")

# Save mono audio
mono_path = os.path.join(nemo_temp_path, "mono_file.wav")
torchaudio.save(
    mono_path,
    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
    16000,
    channels_first=True,
)

update_progress(args.job_id, "diarization_model_loading")
print("Diarization model loading")
# Initialize NeMo model
msdd_model = NeuralDiarizer(cfg=create_config(nemo_temp_path)).to(args.device)

update_progress(args.job_id, "diarization_processing")
print("Diarization processing")

# Run diarization
msdd_model.eval()
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Process RTTM file
rttm_path = os.path.join(nemo_temp_path, "pred_rttms", "mono_file.rttm")

# Process speaker timestamps
speaker_ts = []
speaker_id_map = {}
next_id = 1

with open(rttm_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        original_id = int(line_list[11].split("_")[-1])

        if original_id not in speaker_id_map:
            speaker_id_map[original_id] = next_id
            next_id += 1

        new_id = speaker_id_map[original_id]
        speaker_ts.append([s, e, new_id])

update_progress(args.job_id, "diarization_completed")
print("Diarization completed")

# Finalizing stage
update_progress(args.job_id, "finalizing_started")
print("Finalizing started")

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

# Modified punctuation restoration section
if info.language in punct_model_langs:
    update_progress(args.job_id, "punctuation_restoration")
    print("Punctuation restoration")

    try:
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        # Process words in chunks to be more memory efficient
        words_list = list(map(lambda x: x["word"], wsm))

        labeled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    except Exception as e:
        logging.error(f"Error in punctuation restoration: {str(e)}")
        logging.warning(
            "Continuing with original punctuation due to error in restoration"
        )

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

update_progress(args.job_id, "generating_transcript")
print("Generating transcript")
# Final processing
wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# Capture the transcript in memory first
transcript_io = io.StringIO()
srt_io = io.StringIO()

# Generate speaker-aware transcript and SRT content
get_speaker_aware_transcript(ssm, transcript_io)
write_srt(ssm, srt_io)

# Retrieve the content from in-memory objects
transcript_txt = transcript_io.getvalue()
srt_txt = srt_io.getvalue()

# Apply the cleanup functions
formatted_transcript_txt = format_transcript(transcript_txt)
formatted_srt_txt = format_transcript(srt_txt)

update_progress(args.job_id, "saving_files")
print("Saving files")
# Write transcript output to the specified directory
transcript_output_path = os.path.join(
    args.temp_dir, f"{os.path.splitext(os.path.basename(args.audio))[0]}.txt"
)
with open(transcript_output_path, "w", encoding="utf-8-sig") as f:
    f.write(formatted_transcript_txt)

# Write SRT output to the specified directory
srt_output_path = os.path.join(
    args.temp_dir, f"{os.path.splitext(os.path.basename(args.audio))[0]}.srt"
)
with open(srt_output_path, "w", encoding="utf-8-sig") as f:
    f.write(formatted_srt_txt)

update_progress(args.job_id, "completed")
print("Completed")
# Clean up temporary files
cleanup(temp_outputs)
