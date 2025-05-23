import argparse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from datetime import timedelta
from pathlib import Path
import logging

from moviepy import VideoFileClip
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 16000


def extract_audio(video_path: Path) -> Path:
    """
    Mengekstrak audio dari file video dan menyimpannya sebagai file WAV.
    Parameters
    ----------
    video_path : Path
        Path ke video input.

    Returns
    -------
    Path
        Path ke file audio yang diekstrak (.wav).
    """
    logger.info(f"Mengekstrak audio dari {video_path}...")
    audio_path = video_path.with_suffix(".wav")
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path), codec='pcm_s16le', fps=EXPECTED_SAMPLE_RATE)
    logger.info(f"Audio disimpan ke {audio_path}")
    return audio_path


def load_audio(audio_path: Path):
    """
    Muat file audio dan ubah ke mono 16kHz.

    Parameters
    ----------
    audio_path : Path
        Path ke file audio.

    Returns
    -------
    waveform : torch.Tensor
        Waveform yang dimuat dalam mono.
    sample_rate : int
        Sample rate (16000)/16 kHz.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except RuntimeError:
        logger.warning("Torchaudio gagal, fallback ke soundfile...")
        waveform, sample_rate = sf.read(str(audio_path))
        waveform = torch.tensor(waveform)
        waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != EXPECTED_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, EXPECTED_SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform, EXPECTED_SAMPLE_RATE


def segment_audio(waveform, sample_rate, segment_duration=5.0):
    """
    Memisahkan waveform menjadi segmen overlap.

    Parameters
    ----------
    waveform : torch.Tensor
    sample_rate : int
    segment_duration : float

    Returns
    -------
    List[Dict]
        List segmen dengan waveform dan timestamp.
    """
    samples_per_segment = int(segment_duration * sample_rate)
    overlap_samples = int(0.5 * sample_rate)
    total_samples = waveform.shape[1]

    segments = []
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + samples_per_segment, total_samples)
        segment = waveform[:, start_sample:end_sample]
        segments.append({
            "waveform": segment,
            "start_time": start_sample / sample_rate,
            "end_time": end_sample / sample_rate
        })
        start_sample += samples_per_segment - overlap_samples

    return segments


def format_timestamp(seconds: float) -> str:
    """
    Format detik ke timestamp SRT.

    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
        Timestamp "HH:MM:SS,mmm".
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def transcribe_segments(segments, processor, model, device):
    """
    Mengtranskripsi daftar segmen audio dengan Whisper.

    Parameters
    ----------
    segments : List[Dict]
    processor : WhisperProcessor
    model : WhisperForConditionalGeneration

    Returns
    -------
    List[Dict]
        Hasil transkripsi dengan timestamp.
    """
    logger.info("Mengtranskripsi segmen audio...")
    transcribed = []

    for i, seg in enumerate(segments):
        logger.info(f"Segmen {i + 1}/{len(segments)}")
        # Pindahkan tensor waveform ke CPU/GPU sebelum mengonversi ke numpy
        input_features = processor(
            seg["waveform"].cpu().numpy()[0],
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(device)

        predicted_ids = model.generate(input_features)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        if text:
            transcribed.append({
                "index": i + 1,
                "start": seg["start_time"],
                "end": seg["end_time"],
                "text": text
            })

    return transcribed


def generate_srt(transcribed_segments, output_path: Path):
    """
    Menulis file SRT dari segmen yang ditranskripsi.

    Parameters
    ----------
    transcribed_segments : List[Dict]
    output_path : Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Menulis subtitle ke {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for seg in transcribed_segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    logger.info("Pembuatan subtitle selesai.")


def generate_subtitles(video_path_str):
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    video_path = Path(video_path_str)
    if not video_path.exists():
        logger.error(f"Video tidak ditemukan: {video_path}")
        return

    audio_path = video_path.with_suffix(".wav")
    srt_path = video_path.with_suffix(".srt")

    if not audio_path.exists():
        audio_path = extract_audio(video_path)
    else:
        logger.info(f"Menggunakan audio yang ada: {audio_path}")

    logger.info("Memuat model Whisper...")
    processor = WhisperProcessor.from_pretrained("cahya/whisper-medium-id")
    model = WhisperForConditionalGeneration.from_pretrained("cahya/whisper-medium-id").to(device)

    waveform, sample_rate = load_audio(audio_path)
    waveform = waveform.to(device)

    logger.info("Segmenting audio...")
    segments = segment_audio(waveform, sample_rate)

    transcribed = transcribe_segments(segments, processor, model, device)

    generate_srt(transcribed, srt_path)

    logger.info("Pembuatan subtitle selesai.")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"SRT:   {srt_path}")


def main():
    parser = argparse.ArgumentParser(description='Membuat subtitle bahasa Indonesia dari video')
    parser.add_argument('video_path', type=str, help='Path ke file video')
    args = parser.parse_args()

    generate_subtitles(args.video_path)


if __name__ == "__main__":
    main()