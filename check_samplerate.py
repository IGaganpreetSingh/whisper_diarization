import os
import wave
import librosa
import soundfile as sf
from pydub import AudioSegment


def check_audio_samplerate(file_path):
    """
    Check audio sample rate using multiple libraries and print detailed information
    about the audio file.

    Args:
        file_path (str): Path to the audio file

    Returns:
        dict: Dictionary containing sample rates from different libraries
    """
    results = {}
    file_extension = os.path.splitext(file_path)[1].lower()

    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    print("-" * 50)

    try:
        # Method 1: Using librosa
        try:
            sr_librosa = librosa.get_samplerate(file_path)
            results["librosa"] = sr_librosa
            print(f"Librosa Sample Rate: {sr_librosa} Hz")

            # Get additional info using librosa
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Duration: {duration:.2f} seconds")
            print(f"Number of channels: {1 if len(y.shape) == 1 else y.shape[0]}")
        except Exception as e:
            print(f"Librosa error: {str(e)}")

        # Method 2: Using soundfile
        try:
            sf_info = sf.info(file_path)
            results["soundfile"] = sf_info.samplerate
            print(f"\nSoundFile Information:")
            print(f"Sample Rate: {sf_info.samplerate} Hz")
            print(f"Channels: {sf_info.channels}")
            print(f"Duration: {sf_info.duration:.2f} seconds")
            print(f"Format: {sf_info.format}")
            print(f"Subtype: {sf_info.subtype}")
        except Exception as e:
            print(f"SoundFile error: {str(e)}")

        # Method 3: Using wave (for .wav files only)
        if file_extension == ".wav":
            try:
                with wave.open(file_path, "rb") as wav_file:
                    results["wave"] = wav_file.getframerate()
                    print(f"\nWave Module Information:")
                    print(f"Sample Rate: {wav_file.getframerate()} Hz")
                    print(f"Channels: {wav_file.getnchannels()}")
                    print(f"Sample Width: {wav_file.getsampwidth()} bytes")
                    print(f"Frames: {wav_file.getnframes()}")
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    print(f"Duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Wave error: {str(e)}")

        # Method 4: Using pydub (supports multiple formats)
        try:
            audio = AudioSegment.from_file(file_path)
            results["pydub"] = audio.frame_rate
            print(f"\nPydub Information:")
            print(f"Sample Rate: {audio.frame_rate} Hz")
            print(f"Channels: {audio.channels}")
            print(f"Sample Width: {audio.sample_width} bytes")
            print(f"Duration: {len(audio)/1000:.2f} seconds")
        except Exception as e:
            print(f"Pydub error: {str(e)}")

    except Exception as e:
        print(f"General error: {str(e)}")

    return results


def batch_check_audio_files(directory, extensions=[".wav", ".mp3", ".flac", ".ogg"]):
    """
    Check all audio files in a directory and its subdirectories.

    Args:
        directory (str): Path to the directory containing audio files
        extensions (list): List of audio file extensions to check
    """
    print(f"Checking audio files in: {directory}")
    print("=" * 50)

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                check_audio_samplerate(file_path)


# Example usage
if __name__ == "__main__":
    # Check a single file
    file_path = "E:\Clean Audio.mp3"
    results = check_audio_samplerate(file_path)

    # # Check all audio files in a directory
    # directory = "path/to/your/audio/directory"
    # batch_check_audio_files(directory)
