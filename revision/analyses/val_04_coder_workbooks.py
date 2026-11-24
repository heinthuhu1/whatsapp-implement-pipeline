"""Revision R2 #18: Excel coding workbooks, one per coder, for the extended validation design.

Design (extends val_02_draw_sample.py)
  * the 260-item stratified sample (validation_sample.csv), plus
  * a census of every content-bearing message flagged by the peer-support or workaround
    lexicon that is not already in the sample, so that lexicon precision can be computed
    exactly over all flagged messages, plus
  * a topical-coverage code (five programme components) so the topic keyword lists are
    validated alongside the other lexicons.
  All items are shuffled together under neutral IDs, so coders cannot tell sampled items from
  census items, and no model output or keyword flag is shown.
  A separate practice sheet holds 15 items from outside the sample and census for calibration.

Input : data/Jheeng/revision/clinical_repro/message_level.csv, validation_sample.csv
Output: data/Jheeng/revision/clinical_repro/coding/
          coding_<name>.xlsx   one workbook per coder (instructions, practice, coding, lists)
          coding_key.csv       analyst key: item_id -> msg_id, source, sample weights, flags,
                               model outputs (never give this to coders)

Usage: venv/bin/python revision/analyses/val_04_coder_workbooks.py "Coder One" "Coder Two"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "Jheeng" / "revision" / "clinical_repro"
OUT = SRC / "coding"
SEED = 20261112
TYPES = ["voice_note", "free_text", "case_report_text"]
TYPE_LABEL = {"voice_note": "Voice note", "free_text": "Free text", "case_report_text": "Case report"}
TOPICS = ["training", "coordination", "clinical_protocol", "equipment", "supervision"]

# (column header, list name or None, width)
CODES = [
    ("sentiment", "sentiment", 12),
    ("urgency", "yn", 9),
    ("peer_support", "yn", 9),
    ("workaround", "yn", 9),
    ("hedging", "yn", 9),
    ("translation_adequacy", "adequacy", 11),
    ("language", "language", 13),
] + [("topic_" + t, "yn", 10) for t in TOPICS] + [("notes", None, 30)]
LISTS = {
    "sentiment": ["negative", "neutral", "positive"],
    "yn": ["y", "n"],
    "adequacy": ["1", "2", "3"],
    "language": ["english", "roman_urdu", "urdu_script", "mixed", "other"],
    "language_voice": ["urdu", "english", "mixed", "other"],
}

INSTRUCTIONS = [
    ("Coding workbook for the WhatsApp validation study (reviewer comment R2 #18)", True),
    ("", False),
    ("Read CODING_GUIDE.md in full before starting. This sheet is a summary only.", False),
    ("", False),
    ("How to work", True),
    ("1. Code the 'Practice' sheet first, together with the other coder, and agree any clarifications. Record them in CODING_GUIDE.md.", False),
    ("2. Then code every row of the 'Coding' sheet on your own. Do not discuss items with the other coder until both of you have finished.", False),
    ("3. Pick values from the drop-down lists. Every code column must be filled for every row, except translation_adequacy, which is for voice notes only.", False),
    ("4. Empty code cells are shaded yellow, so you can see what is left. The progress count below updates as you go.", False),
    ("5. Code the meaning of the original message. For voice notes, judge from the Urdu transcript or, where possible, the audio file named in the row; then rate the English translation separately.", False),
    ("", False),
    ("Codes (short definitions; full definitions in the guide)", True),
    ("sentiment: negative / neutral / positive tone as the sender intended it. Factual reports are neutral; clinical severity alone is not negative.", False),
    ("urgency: y if the message asks for or signals an immediate, time-critical response.", False),
    ("peer_support: y if it praises, thanks, encourages, reassures or shows solidarity with a colleague or the team.", False),
    ("workaround: y if it describes doing something differently from the expected procedure because of a barrier.", False),
    ("hedging: y if the sender expresses uncertainty about a fact, decision or plan.", False),
    ("translation_adequacy (voice notes only): 3 adequate, 2 partly adequate, 1 inadequate.", False),
    ("language: the language of the original message. The drop-down differs by row:", False),
    ("   Text messages and case reports (how it is WRITTEN): english / roman_urdu (Urdu in Latin letters, e.g. 'mareez ko hospital le gaye') / urdu_script (Urdu in Urdu letters) / mixed / other.", False),
    ("   Voice notes (what language is SPOKEN): urdu / english / mixed / other (Punjabi, Pahari, Saraiki, Pashto or no speech). Most voice notes will simply be urdu.", False),
    ("   mixed = whole phrases or sentences in two languages. English loanwords inside an Urdu sentence (patient, hospital, ambulance, BP) do NOT make it mixed.", False),
    ("   Why: the sentiment model reads English only, and voice notes were transcribed as Urdu, so accuracy is checked separately by language.", False),
    ("topic_training: y if the message is about training, teaching, drills, refresher sessions or practising skills.", False),
    ("topic_coordination: y if it arranges or directs a response: dispatching a vehicle, pickups, handovers, who goes where, or contacting callers or facilities to organise a case.", False),
    ("topic_clinical_protocol: y if it refers to clinical procedures or protocols: triage, vital signs, treatment steps, SOPs or checklists.", False),
    ("topic_equipment: y if it is about ambulance equipment or supplies (oxygen, stretcher, kits, monitors, the vehicle's equipment), including whether it works.", False),
    ("topic_supervision: y if it is about oversight of staff work: a supervisor or in-charge checking, reviewing, monitoring or instructing.", False),
    ("A message can have several topics, or none. Code the topic from the content, not from single words: 'call me back' about a personal matter is not coordination.", False),
    ("", False),
    ("Data protection", True),
    ("This file contains real programme messages. Keep it on a project-controlled device. Do not upload it to Google Sheets, OneDrive sharing links, translation tools or AI tools. Do not copy patient details into the notes. Return the file to the analyst and delete your copy when done.", False),
]


def clean(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return ILLEGAL_CHARACTERS_RE.sub("", str(x))


def setcell(ws, row, col, value):
    c = ws.cell(row=row, column=col)
    c.value = value
    if isinstance(value, str) and value.startswith("="):
        c.data_type = "s"          # never let message text become a formula
    return c


def build_items():
    m = pd.read_csv(SRC / "message_level.csv")
    s = pd.read_csv(SRC / "validation_sample.csv")
    content = m[m["msg_type"].isin(TYPES)]
    content = content[~((content["msg_type"] == "voice_note") & content["translation_en"].isna())].copy()

    census = content[(content["flag_peer_support"] == 1) | (content["flag_workaround"] == 1)]
    census = census[~census["msg_id"].isin(s["msg_id"])]

    rest = content[~content["msg_id"].isin(s["msg_id"]) & ~content["msg_id"].isin(census["msg_id"])]
    rng = np.random.default_rng(SEED)
    practice = pd.concat([
        rest[rest["msg_type"] == t].sample(n=k, random_state=int(rng.integers(1e9)))
        for t, k in (("voice_note", 6), ("free_text", 6), ("case_report_text", 3))
    ])

    def text_cols(d):
        d = d.copy()
        d["original_text"] = np.where(d["msg_type"] == "voice_note", d["transcription_ur"], d["message"])
        d["english_translation"] = np.where(d["msg_type"] == "voice_note", d["translation_en"], "")
        return d

    census = text_cols(census)
    practice = text_cols(practice)
    msg_cols = ["msg_id", "timestamp", "phase", "msg_type", "vn_filename", "original_text", "english_translation"]
    flag_cols = ["flag_urgency", "flag_hedging", "flag_peer_support", "flag_workaround"] + \
                ["flag_toc_" + t for t in TOPICS] + ["flag_toc_any"]

    a = s[msg_cols + flag_cols + ["item_id", "stratum", "substratum", "inclusion_prob", "design_weight",
                                  "model_label", "model_prob", "model_score"]].copy()
    a = a.rename(columns={"item_id": "sample_item_id"})
    a["source"] = "sample"
    b = census.rename(columns={"label_linked": "model_label", "prob_linked": "model_prob",
                               "score_linked": "model_score"})[msg_cols + flag_cols + ["model_label", "model_prob", "model_score"]].copy()
    b["source"] = "census"
    main = pd.concat([a, b], ignore_index=True)
    main = main.sample(frac=1, random_state=SEED).reset_index(drop=True)
    main.insert(0, "item_id", [f"C{i + 1:03d}" for i in range(len(main))])

    practice = practice[msg_cols].reset_index(drop=True)
    practice.insert(0, "item_id", [f"P{i + 1:02d}" for i in range(len(practice))])
    practice["source"] = "practice"
    return main, practice


def write_sheet(wb, title, d, coder):
    ws = wb.create_sheet(title)
    base = [("item_id", 8), ("phase", 7), ("message_type", 12), ("original_text", 60),
            ("english_translation", 60), ("audio_file", 22)]
    headers = base + [(c, w) for c, _, w in CODES]
    head_fill = PatternFill("solid", fgColor="DDE4EE")
    code_fill = PatternFill("solid", fgColor="EAF3E6")
    for j, (h, w) in enumerate(headers, start=1):
        c = setcell(ws, 1, j, h)
        c.font = Font(bold=True)
        c.fill = code_fill if j > len(base) else head_fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[c.column_letter].width = w
    for i, r in enumerate(d.itertuples(index=False), start=2):
        orig = clean(r.original_text).strip() or ("[no Urdu transcript: judge from the audio or the translation]"
                                                  if r.msg_type == "voice_note" else "[no text in the export]")
        vals = [r.item_id, clean(r.phase), TYPE_LABEL[r.msg_type], orig,
                clean(r.english_translation) if r.msg_type == "voice_note" else "",
                clean(r.vn_filename) if r.msg_type == "voice_note" else ""]
        for j, v in enumerate(vals, start=1):
            c = setcell(ws, i, j, v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for j in range(len(base) + 1, len(headers) + 1):
            ws.cell(row=i, column=j).alignment = Alignment(vertical="top", wrap_text=True)
        if r.msg_type != "voice_note":
            ws.cell(row=i, column=len(base) + 6).fill = PatternFill("solid", fgColor="EEEEEE")   # adequacy n/a
    n = len(d) + 1
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(headers)).column_letter}{n}"
    # drop-downs
    for k, (h, lst, _) in enumerate(CODES):
        if not lst:
            continue
        col = ws.cell(row=1, column=len(base) + 1 + k).column_letter
        def mkdv(name):
            L = chr(65 + list(LISTS).index(name))
            v = DataValidation(type="list", formula1=f"=Lists!${L}$2:${L}${1 + len(LISTS[name])}",
                               allow_blank=True, showErrorMessage=True,
                               errorTitle="Invalid code", error="Pick a value from the list.")
            ws.add_data_validation(v)
            return v
        if lst == "language":
            dv_text, dv_voice = mkdv("language"), mkdv("language_voice")
            for i, r in enumerate(d.itertuples(index=False), start=2):
                (dv_voice if r.msg_type == "voice_note" else dv_text).add(f"{col}{i}")
        else:
            mkdv(lst).add(f"{col}2:{col}{n}")
    # yellow shading for empty required cells
    yellow = PatternFill("solid", fgColor="FFF2B3")
    adequacy_col = ws.cell(row=1, column=len(base) + 6).column_letter
    for k, (h, lst, _) in enumerate(CODES):
        if not lst:
            continue
        col = ws.cell(row=1, column=len(base) + 1 + k).column_letter
        if col == adequacy_col:
            rule = FormulaRule(formula=[f'AND($C2="Voice note",ISBLANK({col}2))'], fill=yellow)
        else:
            rule = FormulaRule(formula=[f"ISBLANK({col}2)"], fill=yellow)
        ws.conditional_formatting.add(f"{col}2:{col}{n}", rule)
    for i in range(2, n + 1):
        ws.row_dimensions[i].height = 75
    return ws


def build_workbook(coder, main, practice, path):
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 130
    setcell(ins, 1, 1, f"Coder: {coder}").font = Font(bold=True, size=13)
    for i, (t, bold) in enumerate(INSTRUCTIONS, start=3):
        c = setcell(ins, i, 1, t)
        c.font = Font(bold=bold, size=12 if bold else 11)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    r = len(INSTRUCTIONS) + 4
    setcell(ins, r, 1, "Progress").font = Font(bold=True, size=12)
    n = len(main) + 1
    setcell(ins, r + 1, 1, f'="Coding sheet: "&COUNTA(Coding!G2:G{n})&" of {len(main)} items have a sentiment code; "'
                           f'&(COUNTBLANK(Coding!G2:K{n})+COUNTBLANK(Coding!M2:R{n})+COUNTIFS(Coding!C2:C{n},"Voice note",Coding!L2:L{n},""))&" required code cells are still empty."').data_type = "f"
    write_sheet(wb, "Practice", practice, coder)
    write_sheet(wb, "Coding", main, coder)
    lists = wb.create_sheet("Lists")
    for j, (name, vals) in enumerate(LISTS.items(), start=1):
        setcell(lists, 1, j, name).font = Font(bold=True)
        for i, v in enumerate(vals, start=2):
            setcell(lists, i, j, v)
    lists.sheet_state = "hidden"
    wb.active = wb.sheetnames.index("Instructions")
    wb.save(path)


def main() -> None:
    coders = sys.argv[1:] or ["Coder 1", "Coder 2"]
    OUT.mkdir(exist_ok=True)
    main_items, practice = build_items()
    key = pd.concat([main_items, practice], ignore_index=True)
    key.to_csv(OUT / "coding_key.csv", index=False)
    for c in coders:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_")
        build_workbook(c, main_items, practice, OUT / f"coding_{slug}.xlsx")
    print(f"coding items: {len(main_items)} ({(main_items.source == 'sample').sum()} sample, "
          f"{(main_items.source == 'census').sum()} census); practice items: {len(practice)}")
    print(main_items.groupby(["source", "msg_type"]).size().to_string())
    print("flagged in coding set: peer support", int(main_items.flag_peer_support.sum()),
          "workaround", int(main_items.flag_workaround.sum()))
    print("written to", OUT)


if __name__ == "__main__":
    main()
