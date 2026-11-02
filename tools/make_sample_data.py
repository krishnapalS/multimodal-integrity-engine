#!/usr/bin/env python3
"""Generate the synthetic evaluation corpus in ../sample_submissions/.

13 files (5 PDF, 5 DOCX, 3 MP4) with deliberately embedded relationships,
plus dataset_manifest.json recording the ground truth (brief section 7).
MP4 speech is synthesized with gTTS (Google TTS, free) and muxed onto a
black video with ffmpeg; the silent clip is generated with no audio stream.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from case_texts import (AI_ESSAY, DRONE_ANALYSIS, EDTECH_ANALYSIS,
                        FARMFRESH_ANALYSIS, GREENCHARGE_ANALYSIS,
                        GREENCHARGE_PARAPHRASE, HOSPITAL_ANALYSIS,
                        MP4_CROSSMODAL_SPEECH, MP4_ORIGINAL_SPEECH,
                        VERTICAL_EV_ANALYSIS)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_submissions")
HEADER = "Unstop Case Challenge 2026 | Candidate Submission"
FOOTER = "Confidential - For Evaluation Only"


def write_pdf(path: str, text: str, title: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer)

    styles = getSampleStyleSheet()

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(72, letter[1] - 40, HEADER)
        canvas.drawCentredString(letter[0] / 2, 30, f"{FOOTER} - page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=letter,
                            leftMargin=72, rightMargin=72,
                            topMargin=72, bottomMargin=72)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    paras = [" ".join(p.split()) for p in text.split("\n\n")]
    paras = [p for p in paras if p]
    for i, para in enumerate(paras):
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 8))
        # Force a second page so repeated header/footer furniture exists
        # for the extractor's stripping logic to detect.
        if i == max(0, len(paras) // 2 - 1):
            story.append(PageBreak())
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def write_docx(path: str, text: str, title: str):
    import docx
    document = docx.Document()
    document.add_heading(title, level=1)
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if para:
            document.add_paragraph(para)
    document.save(path)


def tts_to_mp3(text: str, mp3_path: str):
    from gtts import gTTS
    gTTS(text=text, lang="en", tld="co.in").save(mp3_path)


def make_talk_mp4(text: str, out_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "speech.mp3")
        tts_to_mp3(text, mp3)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=640x360:r=10",
            "-i", mp3, "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            out_path,
        ], check=True)


def make_silent_mp4(out_path: str, seconds: int = 12):
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=0x202020:s=640x360:r=10:d={seconds}",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
    ], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [
        ("submission_001_greencharge.pdf", write_pdf, GREENCHARGE_ANALYSIS,
         "GreenCharge Mobility - Market Entry Analysis"),
        ("submission_002_farmfresh.pdf", write_pdf, FARMFRESH_ANALYSIS,
         "GreenCharge Mobility - Operations Review"),
        ("submission_003_drones.pdf", write_pdf, DRONE_ANALYSIS,
         "Urban Drone Logistics - Pilot Design"),
        ("submission_004_vertical_ev.pdf", write_pdf, VERTICAL_EV_ANALYSIS,
         "Battery Supply Chain Strategy"),
    ]
    for name, fn, text, title in jobs:
        fn(os.path.join(OUT, name), text, title)
        print("wrote", name)

    open(os.path.join(OUT, "submission_005_corrupted.pdf"), "wb").close()
    print("wrote submission_005_corrupted.pdf (0 bytes)")

    docx_jobs = [
        ("submission_006_greencharge_reworded.docx", GREENCHARGE_PARAPHRASE,
         "GreenCharge Expansion Strategy"),
        ("submission_007_drones_copy.docx", DRONE_ANALYSIS,
         "Urban Drone Logistics - Pilot Design"),
        ("submission_008_quickcommerce.docx", AI_ESSAY,
         "The Future of Quick Commerce in India"),
        ("submission_009_hospital.docx", HOSPITAL_ANALYSIS,
         "Outpatient Queue Optimization"),
        ("submission_010_edtech.docx", EDTECH_ANALYSIS,
         "Gamified Learning Progression Design"),
    ]
    for name, text, title in docx_jobs:
        write_docx(os.path.join(OUT, name), text, title)
        print("wrote", name)

    make_talk_mp4(MP4_CROSSMODAL_SPEECH, os.path.join(OUT, "submission_011_pitch.mp4"))
    print("wrote submission_011_pitch.mp4")
    make_silent_mp4(os.path.join(OUT, "submission_012_silent.mp4"))
    print("wrote submission_012_silent.mp4 (no audio)")
    make_talk_mp4(MP4_ORIGINAL_SPEECH, os.path.join(OUT, "submission_013_original_pitch.mp4"))
    print("wrote submission_013_original_pitch.mp4")


if __name__ == "__main__":
    main()
