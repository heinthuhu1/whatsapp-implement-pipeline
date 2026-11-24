#!/usr/bin/env python3
"""Convert a WhatsApp chat that was pasted into a .docx back into a faithful
WhatsApp .txt export the pipeline can parse.

Handles:
  - one-paragraph-per-line docx structure (joins paragraphs with newlines)
  - <w:br/> / <w:tab/> inside a paragraph
  - Word word-wrap that splits media markers, e.g. "...opus (file" + "attached)"
  - narrow no-break space (U+202F) and NBSP (U+00A0) in timestamps -> normal space

Usage:
  python tools/docx_to_whatsapp.py "input.docx" data/raw/output_chat.txt
"""
from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path

TOKEN_RE = re.compile(
    r"<w:t[^>]*>(.*?)</w:t>|<w:br\s*/?>|<w:cr\s*/?>|<w:tab\s*/?>", re.S
)


def paragraph_text(p_xml: str) -> str:
    """Reconstruct a paragraph's text, preserving in-run breaks/tabs in order."""
    out: list[str] = []
    for m in TOKEN_RE.finditer(p_xml):
        tag = m.group(0)
        if tag.startswith("<w:t") and not tag.startswith("<w:tab"):
            out.append(html.unescape(m.group(1)))
        elif tag.startswith("<w:tab"):
            out.append("\t")
        else:  # <w:br/> or <w:cr/>
            out.append("\n")
    return "".join(out)


def docx_to_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    paras = [paragraph_text(p) for p in re.split(r"</w:p>", xml)]
    text = "\n".join(paras)

    # Normalise exotic whitespace so the timestamp parser matches.
    text = text.replace(" ", " ").replace(" ", " ")

    # Repair Word word-wrap that split WhatsApp media markers across lines.
    text = re.sub(r"\(file\s*\n\s*attached\)", "(file attached)", text)
    text = re.sub(r"<Media\s*\n\s*omitted>", "<Media omitted>", text)

    # Drop blank lines introduced by docx paragraph spacing (real WhatsApp
    # exports have none). Non-timestamped content lines are kept and become
    # message continuations during parsing.
    lines = [ln for ln in text.split("\n") if ln.strip() != ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = docx_to_text(src)
    dst.write_text(text, encoding="utf-8")

    ts = re.compile(r"^\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s*(?:am|pm)?\s*-\s", re.I)
    lines = text.splitlines()
    n_msgs = sum(1 for ln in lines if ts.match(ln))
    print(f"[docx_to_whatsapp] wrote {dst}")
    print(f"  total lines: {len(lines)} | timestamped messages: {n_msgs}")
    n_cases = len(re.findall(r"(?im)^\s*case\s*no", text))
    print(f"  .opus refs: {text.count('.opus')} | 'Case No' lines: {n_cases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
