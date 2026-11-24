"""Revision R2 #22: run the unmodified pipeline stages on the synthetic export.

Creates a self-contained run directory, revision/reproducibility/synthetic_outputs/, laid out as the
pipeline expects (config/settings.yaml, data/raw/whatsapp_export.txt), and runs each stage of src/
as a subprocess with that directory as the working directory. Because every stage resolves its
paths relative to the working directory, nothing outside synthetic_outputs/ is read or written
(apart from the read-only source files in src/ and the local Hugging Face model cache).

Stage 02 (hosted Whisper transcription) is never run: OPENAI_API_KEY is removed from the
environment and the stage is skipped. Its output format is demonstrated instead by copying
synthetic_voice_notes.csv to data/processed/voice_notes.csv.

The sentiment model is loaded with HF_HUB_OFFLINE=1. On a fresh machine, download it once
(see README.md) or unset HF_HUB_OFFLINE.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "revision" / "reproducibility"
RUN = PKG / "synthetic_outputs"
STAGES = ["00_anonymise", "01_parse", "02_voice_notes", "03_engagement", "04_network",
          "05_sentiment", "06_fidelity", "07_triangulation", "08_visualise", "09_table1"]


def main() -> None:
    if RUN.exists():
        shutil.rmtree(RUN)
    (RUN / "config").mkdir(parents=True)
    (RUN / "data" / "raw").mkdir(parents=True)
    (RUN / "data" / "processed").mkdir(parents=True)
    shutil.copy(PKG / "synthetic_settings.yaml", RUN / "config" / "settings.yaml")
    shutil.copy(PKG / "synthetic_whatsapp_export.txt", RUN / "data" / "raw" / "whatsapp_export.txt")
    shutil.copy(PKG / "synthetic_voice_notes.csv", RUN / "data" / "processed" / "voice_notes.csv")

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg"})
    py = sys.executable
    status, log = [], []
    for st in STAGES:
        if st == "02_voice_notes":
            status.append({"stage": st, "status": "skipped (hosted API; synthetic voice_notes.csv supplied)",
                           "seconds": 0})
            continue
        t0 = time.time()
        r = subprocess.run([py, str(ROOT / "src" / f"{st}.py")], cwd=RUN, env=env,
                           capture_output=True, text=True)
        ok = r.returncode == 0
        status.append({"stage": st, "status": "ok" if ok else f"FAILED (exit {r.returncode})",
                       "seconds": round(time.time() - t0, 1)})
        noise = ("Warning", "warn(", "Loading weights", "it/s]")
        err = "\n".join(l for l in r.stderr.splitlines() if not any(n in l for n in noise))
        log.append(f"===== {st} (exit {r.returncode})\n{r.stdout}{err}\n")
        print(f"{st}: {'ok' if ok else 'FAILED'}")
    pd.DataFrame(status).to_csv(RUN / "stage_status.csv", index=False)
    (RUN / "run_log.txt").write_text("\n".join(log))

    # quick summary of what the run produced
    msgs = pd.read_csv(RUN / "data" / "interim" / "messages.csv")
    cases = pd.read_csv(RUN / "data" / "interim" / "case_reports.csv")
    summary = [
        f"messages parsed: {len(msgs)}; by type: {msgs['message_type'].value_counts().to_dict()}",
        f"senders: {msgs['sender_code'].nunique()}; phases: {msgs['phase'].value_counts().to_dict()}",
        f"case reports extracted (after de-duplication): {len(cases)}",
        "files written: " + ", ".join(sorted(str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file())),
    ]
    (RUN / "run_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
