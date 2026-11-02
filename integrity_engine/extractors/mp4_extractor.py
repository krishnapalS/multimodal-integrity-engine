"""MP4 extraction: demux audio with ffmpeg, transcribe with faster-whisper.

Timestamp provenance comes from whisper's segment offsets. Silent videos and
files with no audio stream are flagged, never fatal (brief scenario 3).
Model loading is lazy and the model is freed after each batch of files so
peak RSS stays within small containers.
"""
import os
import subprocess
import tempfile

from integrity_engine.config import Config
from integrity_engine.models import Atom, ExtractedDocument

_CFG = Config()


def _has_audio_stream(path: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
        return "audio" in out.stdout
    except Exception:
        return False


def _extract_wav(path: str, wav_path: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path,
         "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
        capture_output=True, text=True, timeout=300,
    )
    return proc.returncode == 0 and os.path.getsize(wav_path) > 0


def extract_mp4(path: str, whisper_model=None) -> ExtractedDocument:
    doc_id = os.path.basename(path)
    doc = ExtractedDocument(doc_id=doc_id, path=path, fmt="mp4")

    if os.path.getsize(path) == 0:
        doc.status = "empty"
        doc.warnings.append("file is 0 bytes; nothing to extract")
        return doc
    if not _has_audio_stream(path):
        doc.status = "no_speech"
        doc.warnings.append("no audio stream found (silent video)")
        return doc

    if whisper_model is None:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            _CFG.whisper_model, device="cpu", compute_type=_CFG.whisper_compute_type
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        if not _extract_wav(path, wav_path):
            doc.status = "corrupted"
            doc.warnings.append("ffmpeg could not decode an audio track")
            return doc
        try:
            segments, _info = whisper_model.transcribe(
                wav_path, beam_size=1, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 700},
            )
            for seg in segments:
                text = " ".join(seg.text.split())
                if text:
                    doc.atoms.append(
                        Atom(text=text, ts_start=seg.start, ts_end=seg.end)
                    )
        except Exception as exc:
            doc.status = "corrupted"
            doc.warnings.append(f"transcription failed: {exc}")
            return doc

    if not doc.atoms:
        doc.status = "no_speech"
        doc.warnings.append("audio stream present but no speech detected")
    return doc
