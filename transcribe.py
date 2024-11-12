import argparse
import logging
import os
import re
import io
import torch
import torchaudio
import librosa
import soundfile as sf
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

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
from transcription_helpers import transcribe_batched
import random
import numpy as np

def set_seed(seed=42):
    """
    Set seeds for reproducibility with a balance of practicality.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set CUDA deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Instead of strict deterministic algorithms, use warning mode
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)


# Call this function once at the beginning
set_seed()

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
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
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
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
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

args = parser.parse_args()

# Create temp directories
temp_outputs = os.path.join(args.temp_dir, "temp_outputs")
os.makedirs(temp_outputs, exist_ok=True)

# Create NeMo specific directories within temp_outputs
nemo_temp_path = os.path.join(temp_outputs, "nemo")
os.makedirs(nemo_temp_path, exist_ok=True)

language = process_language_arg(args.language, args.model_name)


def enhance_vocals(audio_path, output_path, volume_boost_db=6.0):
    """
    Enhance the isolated vocals:
    1. Apply volume boost
    2. Normalize audio
    3. Apply subtle compression
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Normalize audio first
    audio_norm = librosa.util.normalize(audio)

    # Calculate volume boost factor (converting from dB)
    boost_factor = np.power(10.0, volume_boost_db / 20.0)

    # Apply volume boost
    audio_boosted = audio_norm * boost_factor

    # Apply compression (subtle)
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

    # Final normalization to prevent clipping
    audio_final = librosa.util.normalize(audio_compressed)

    # Save the enhanced audio
    sf.write(output_path, audio_final, sr)
    return output_path


# Update the demucs command to use the job-specific directory
import librosa
import soundfile as sf
import numpy as np


def enhance_vocals(audio_path, output_path, volume_boost_db=6.0):
    """
    Enhance the isolated vocals:
    1. Apply volume boost
    2. Normalize audio
    3. Apply subtle compression
    """
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Normalize audio first
    audio_norm = librosa.util.normalize(audio)

    # Calculate volume boost factor (converting from dB)
    boost_factor = np.power(10.0, volume_boost_db / 20.0)

    # Apply volume boost
    audio_boosted = audio_norm * boost_factor

    # Apply compression (subtle)
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

    # Final normalization to prevent clipping
    audio_final = librosa.util.normalize(audio_compressed)

    # Save the enhanced audio
    sf.write(output_path, audio_final, sr)
    return output_path

# Insert this in your main code after the Demucs separation
if args.stemming:
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "{temp_outputs}"'
    )
    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocals_path = os.path.join(
            temp_outputs,
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
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
else:
    vocal_target = args.audio

# Transcribe the audio file
whisper_results, language, audio_waveform = transcribe_batched(
    vocal_target,
    language,
    args.batch_size,
    args.model_name,
    mtypes[args.device],
    args.suppress_numerals,
    args.device,
)

# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

audio_waveform = (
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device)
)
emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=args.batch_size
)

del alignment_model
torch.cuda.empty_cache()

full_transcript = "".join(segment["text"] for segment in whisper_results)

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)

# Save mono audio for NeMo in the NeMo-specific temp directory
mono_path = os.path.join(nemo_temp_path, "mono_file.wav")
torchaudio.save(
    mono_path,
    audio_waveform.cpu().unsqueeze(0).float(),
    16000,
    channels_first=True,
)

# Initialize NeMo MSDD diarization model with updated config path
msdd_model = NeuralDiarizer(cfg=create_config(nemo_temp_path)).to(args.device)
msdd_model.eval()
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Update RTTM file path to use NeMo temp directory
rttm_path = os.path.join(nemo_temp_path, "pred_rttms", "mono_file.rttm")

# Reading timestamps <> Speaker Labels mapping
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

# Modified get_words_speaker_mapping function call
wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labeled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
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

else:
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

# Punctuation restoration and realignment
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
formatted_srt_txt = format_transcript(srt_txt)  # You'll need to implement this function

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

# Clean up temporary files
cleanup(temp_outputs)
