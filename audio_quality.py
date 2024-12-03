import librosa
import numpy as np
import ffmpeg
import os
from typing import Dict, Tuple


class UniversalAudioAnalyzer:
    def __init__(self, frame_length_ms: int = 30):
        self.frame_length_ms = frame_length_ms
        self.supported_direct_formats = {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
        }

    def _load_audio_ffmpeg(
        self, file_path: str, target_sr: int = 44100
    ) -> Tuple[np.ndarray, int]:
        try:
            probe = ffmpeg.probe(file_path)
            audio_info = next(s for s in probe["streams"] if s["codec_type"] == "audio")

            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(
                stream,
                "pipe:",
                format="f32le",
                acodec="pcm_f32le",
                ac=1,
                ar=str(target_sr),
            )

            out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            audio_data = np.frombuffer(out, np.float32)

            return audio_data, target_sr

        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            raise

    def analyze_file(self, file_path: str) -> Dict[str, float]:
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in self.supported_direct_formats:
                try:
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                except:
                    audio_data, sample_rate = self._load_audio_ffmpeg(file_path)
            else:
                audio_data, sample_rate = self._load_audio_ffmpeg(file_path)

            return self.analyze_audio(audio_data, sample_rate)

        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")
            return {"error": str(e), "status": "failed"}

    def _get_active_speech_level(self, signal: np.ndarray, fs: int) -> float:
        """ITU-T P.56 based active speech level measurement"""
        time_const = 0.03  # 30ms time constant
        samples_const = int(time_const * fs)

        # Calculate envelope
        envelope = np.abs(signal)

        # Adaptive threshold calculation
        sorted_env = np.sort(envelope)
        background_noise = np.mean(
            sorted_env[: int(len(sorted_env) * 0.1)]
        )  # Bottom 10%
        speech_thresh = max(background_noise * 3, np.mean(envelope) * 0.2)

        # Activity detection with hangover
        speech_mask = np.zeros_like(signal, dtype=bool)
        active = envelope > speech_thresh

        # Apply hangover scheme (extended speech activity)
        for i in range(len(active) - samples_const):
            if any(active[i : i + samples_const]):
                speech_mask[i : i + samples_const] = True

        active_samples = signal[speech_mask]
        return np.sqrt(np.mean(active_samples**2)) if len(active_samples) > 0 else 0

    def _estimate_true_noise(self, signal: np.ndarray, fs: int) -> float:
        """MMSE-based noise estimation"""
        frame_length = int(0.02 * fs)  # 20ms frames
        frames = librosa.util.frame(
            signal, frame_length=frame_length, hop_length=frame_length // 2
        )

        # Initial noise estimate
        noise_estimate = np.mean(
            np.sort(np.mean(frames**2, axis=0))[: int(frames.shape[1] * 0.1)]
        )

        # Recursive noise update
        alpha = 0.95
        noise_power = np.zeros(frames.shape[1])

        for i in range(frames.shape[1]):
            frame_power = np.mean(frames[:, i] ** 2)
            speech_prob = (
                1 - min(noise_estimate / frame_power, 1) if frame_power > 0 else 0
            )

            if speech_prob < 0.5:  # Likely noise frame
                noise_estimate = alpha * noise_estimate + (1 - alpha) * frame_power

            noise_power[i] = noise_estimate

        return np.sqrt(np.mean(noise_power))

    def analyze_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, float]:
        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)

        # Normalize audio
        audio_norm = librosa.util.normalize(audio_data)

        # Calculate accurate speech level and noise level
        speech_level = self._get_active_speech_level(audio_norm, sample_rate)
        noise_level = self._estimate_true_noise(audio_norm, sample_rate)

        # Calculate accurate SNR
        snr = (
            20 * np.log10(speech_level / noise_level)
            if noise_level > 0
            else float("inf")
        )

        # Additional metrics
        peak_level = np.max(np.abs(audio_norm))
        crest_factor = (
            20 * np.log10(peak_level / speech_level) if speech_level > 0 else 0
        )

        # Spectral analysis
        spec = np.abs(librosa.stft(audio_norm))
        spectral_centroid = librosa.feature.spectral_centroid(
            S=spec, sr=sample_rate
        ).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=spec, sr=sample_rate
        ).mean()

        # Clipping detection
        clipping_threshold = 0.99
        samples_clipped = np.sum(np.abs(audio_norm) > clipping_threshold)
        clipping_percentage = (samples_clipped / len(audio_norm)) * 100

        return {
            "noise_floor_db": (
                20 * np.log10(noise_level) if noise_level > 0 else float("-inf")
            ),
            "signal_level_db": (
                20 * np.log10(speech_level) if speech_level > 0 else float("-inf")
            ),
            "peak_level_db": 20 * np.log10(peak_level),
            "snr_db": snr,
            "crest_factor_db": crest_factor,
            "spectral_centroid_hz": float(spectral_centroid),
            "spectral_bandwidth_hz": float(spectral_bandwidth),
            "clipping_percentage": clipping_percentage,
            "duration_seconds": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
        }


def analyze_media_quality(file_path: str) -> Dict[str, float]:
    analyzer = UniversalAudioAnalyzer()
    return analyzer.analyze_file(file_path)
