"""Revision R2 #22: generate a fully synthetic WhatsApp group export for the reproducibility package.

Writes revision/reproducibility/synthetic_whatsapp_export.txt (and a small synthetic voice-note
table, synthetic_voice_notes.csv). Every sender name, place name, case and message is invented by
the templates below; no real data is read. Deterministic (seed fixed), so the file is identical on
every run.

The export imitates an Android WhatsApp export with a day/month/year date and a 12-hour clock:
    "14/11/2024, 9:05 am - Kestrel Synthetic: message text"
and includes multi-line case reports in two template generations (as in the demonstration
programme), voice-note and media placeholders, deleted-message notices and group system notices.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "revision" / "reproducibility"
SEED = 22
START = datetime(2024, 11, 4, 8, 0)
END = datetime(2025, 2, 28, 22, 0)

# Obviously fictional senders (bird code names with a shared fictional surname)
SENDERS = {
    "coord": "Kestrel Synthetic",     # coordinator, posts most messages
    "sup": "Oriole Synthetic",        # supervisor
    "emp1": "Heron Synthetic",
    "emp2": "Plover Synthetic",
    "emp3": "Wren Synthetic",
    "emp4": "Lark Synthetic",
    "drv": "Falcon Synthetic",        # driver
    "trn": "Egret Synthetic",         # trainer, joins later
}
WEIGHTS = {"coord": 30, "sup": 10, "emp1": 14, "emp2": 12, "emp3": 8, "emp4": 6, "drv": 6, "trn": 3}
VILLAGES = ["Village Alpha", "Village Bravo", "Hamlet Charlie", "Village Delta", "Hamlet Echo"]
HOSPITALS = ["District Hospital Demo", "Rural Health Centre Demo", "Teaching Hospital Demo", "No hospital"]
COMPLAINTS = ["fever", "Fever", "delivery", "head injury", "foot injury", "fever and vomiting",
              "chest pain", "asthma", "leg fracture", "kidney problem", "stomach pain", "dog bite",
              "unconscious", "BP high", "appointment", "dead body", "burn", "fell from roof"]
GENDERS = ["female", "Female", "male", "Male", "child", "female child", "male child", "Mal", ""]
FIRST_AID = ["vitals monitoring", "oxygen given", "splint and bandage", "wound dressing",
             "vitals monitoring and positioning", "none"]

ENGLISH = [
    "Good morning team", "Ambulance is ready for duty", "Vehicle checked, fuel is full",
    "Patient shifted to hospital safely", "Please share today's case reports",
    "Well done team, good job on yesterday's call", "Thank you everyone for the quick response",
    "Emergency call from {v}, team leaving now", "Urgent: need a second stretcher at {v}",
    "Maybe we can do the refresher training next week", "I think the oxygen cylinder needs refilling",
    "Not sure if the road to {v} is open", "Road blocked, taking the alternative route instead",
    "Oxygen kit not working, used the spare kit instead", "Supervisor will review the checklist on Friday",
    "Reminder: follow the triage protocol for every patient", "Monthly drill on Saturday at 10am",
    "Please check vitals every 15 minutes during transfer", "Great work on the delivery case",
    "Who is on night duty today?", "Ok", "Done", "Noted", "Received",
    "Dispatch confirmed, ETA 20 minutes", "Stretcher repaired and back in the ambulance",
    "Workshop on first aid SOP next month", "Please call me when you reach {v}",
]
ROMAN_URDU = [
    "Assalam o alaikum sab ko", "Gari tayar hai, fuel bhi full hai", "Patient ko hospital shift kar diya hai",
    "Hum {v} ja rahe hain, patient ki halat theek nahi", "Bhai jaldi aao, emergency hai",
    "Shukriya sab ka, bohat acha kaam kiya", "Mashallah team ne zabardast kaam kiya",
    "Shayad kal training ho gi", "Mujhe lagta hai oxygen kam hai", "Rasta band hai, doosre raste se ja rahe hain",
    "Ji theek hai", "Ji bhai", "Kal subah duty pe aa jaun ga", "Allah behtar kare",
    "Report bhej di hai group mein", "Stretcher ka pahiya kharab hai", "Supervisor sahab kal check karein ge",
    "Aaj raat kaun duty pe hai?", "Patient ke ghar walay saath hain", "Gari wapis station pe aa gayi",
]
URDU_SCRIPT = ["السلام علیکم", "جی ٹھیک ہے", "شکریہ", "مریض کو ہسپتال پہنچا دیا ہے", "ان شاء اللہ"]
VOICE_EN = [
    "We have shifted the patient to the ambulance and are heading to the hospital.",
    "The patient has been handed over at the emergency department.",
    "Road is blocked so we are taking another route.",
    "Please send the second team, this is an emergency.",
    "Thank you all for your hard work today.",
    "The oxygen cylinder is nearly empty, we need a refill.",
    "We reached the house, the patient is stable now.",
    "Maybe the training will be on Monday, I am not sure.",
]
VOICE_UR = [
    "ہم نے مریض کو ایمبولینس میں شفٹ کر دیا ہے اور ہسپتال جا رہے ہیں",
    "مریض کو ایمرجنسی میں حوالے کر دیا ہے",
    "راستہ بند ہے اس لیے دوسرے راستے سے جا رہے ہیں",
    "دوسری ٹیم بھیجیں یہ ایمرجنسی ہے",
    "آج سب کی محنت کا شکریہ",
    "آکسیجن سلنڈر تقریباً خالی ہے",
    "ہم گھر پہنچ گئے ہیں مریض اب ٹھیک ہے",
    "شاید ٹریننگ پیر کو ہو گی مجھے یقین نہیں",
]


def fmt_ts(t: datetime) -> str:
    h = t.hour % 12 or 12
    return f"{t.day:02d}/{t.month:02d}/{t.year}, {h}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'}"


def case_report_v1(n: int, d: datetime, rng: random.Random) -> str:
    t1 = d.replace(hour=rng.randint(7, 20), minute=rng.randint(0, 59))
    t2 = t1 + timedelta(minutes=rng.randint(25, 120))
    hosp = rng.choice(HOSPITALS)
    lines = [
        "🚑🏥",
        f"Date:{d.day}/{d.month}/{d.year}",
        f"Case No:{n:02d}",
        f"Gender: {rng.choice(GENDERS)}",
        f"Initial Complaint: {rng.choice(COMPLAINTS)}",
        f"Location: {rng.choice(VILLAGES)}",
        f"Time patient reached:{t1.strftime('%I:%M%p').lower()}",
        f"First Aid Provided: {rng.choice(FIRST_AID)}",
        f"Name of Hospital Transported to:{hosp}",
        f"Time Hospital was Reached: {t2.strftime('%I:%M%p').lower()}",
        f"EMP Personnel Names: {SENDERS['emp1'].split()[0]} and {SENDERS['emp2'].split()[0]}",
    ]
    return "\n".join(lines)


def case_report_v2(n: int, d: datetime, rng: random.Random) -> str:
    t1 = d.replace(hour=rng.randint(7, 20), minute=rng.randint(0, 59))
    t2 = t1 + timedelta(minutes=rng.randint(25, 120))
    lines = [
        f"*Case no {n}",
        f"Date: {d.day:02d}.{d.month:02d}.{d.year}",
        f"Gender: {rng.choice(GENDERS)}",
        f"Complaint: {rng.choice(COMPLAINTS)}",
        f"Pickup location: {rng.choice(VILLAGES)}",
        f"Pickup time: {t1.strftime('%H:%M')}",
        f"Destination: {rng.choice(HOSPITALS)}",
        f"Arrival time: {t2.strftime('%H:%M')}",
        f"First aid: {rng.choice(FIRST_AID)}",
        f"EMP personnel: {SENDERS['emp3'].split()[0]}, {SENDERS['emp4'].split()[0]}",
    ]
    return "\n".join(lines)


def main() -> None:
    rng = random.Random(SEED)
    events: list[tuple[datetime, str]] = []
    coord = SENDERS["coord"]
    events.append((START - timedelta(minutes=2), "Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more."))
    events.append((START - timedelta(minutes=1), f"{coord} created group \"Demo Ambulance Team\""))
    for k in ("sup", "emp1", "emp2", "emp3", "emp4", "drv"):
        events.append((START, f"{coord} added {SENDERS[k]}"))
    events.append((datetime(2025, 1, 6, 9, 0), f"{coord} added {SENDERS['trn']}"))
    events.append((datetime(2025, 2, 10, 18, 30), f"{SENDERS['emp4']} left"))
    events.append((datetime(2025, 1, 2, 10, 0), f"{coord} changed the group description"))

    senders_active = lambda t: [k for k in SENDERS if not (k == "trn" and t < datetime(2025, 1, 6, 9, 5))
                                and not (k == "emp4" and t > datetime(2025, 2, 10, 18, 30))]

    vn_rows = []
    case_no = 1
    template_switch = datetime(2025, 1, 15)
    day = START
    wa_seq: dict[str, int] = {}
    while day.date() <= END.date():
        # quiet stretch to produce a silence period
        if datetime(2024, 12, 24) <= day <= datetime(2024, 12, 27):
            day += timedelta(days=1)
            continue
        n_msgs = rng.choice([1, 2, 3, 4, 4, 5, 6]) if day.weekday() < 5 else rng.choice([0, 1, 2, 3])
        if day.date() == datetime(2025, 2, 3).date():
            n_msgs = 22  # a busy day to produce a velocity spike
        t = day.replace(hour=rng.randint(7, 10), minute=rng.randint(0, 59))
        for _ in range(n_msgs):
            t += timedelta(minutes=rng.choice([1, 2, 4, 7, 12, 25, 40, 90, 180]))
            if t.date() != day.date():
                break
            act = senders_active(t)
            who = rng.choices(act, weights=[WEIGHTS[k] for k in act])[0]
            name = SENDERS[who]
            r = rng.random()
            if r < 0.22 and who in ("coord", "emp1", "emp2", "emp3"):
                body = case_report_v1(case_no, t, rng) if t < template_switch else case_report_v2(case_no, t, rng)
                events.append((t, f"{name}: {body}"))
                if rng.random() < 0.08:  # repost with a correction, to exercise de-duplication
                    events.append((t + timedelta(minutes=3), f"{name}: {body.replace('Gender: ', 'Gender: female ', 1)}"))
                case_no += 1
            elif r < 0.45:
                key = t.strftime("%Y%m%d")
                wa_seq[key] = wa_seq.get(key, 0) + rng.randint(1, 9)
                fn = f"PTT-{key}-WA{wa_seq[key]:04d}.opus"
                events.append((t, f"{name}: {fn} (file attached)"))
                i = rng.randrange(len(VOICE_EN))
                vn_rows.append({"filename": fn, "date": t.date().isoformat(), "sequence": wa_seq[key],
                                "duration_seconds": round(rng.uniform(3, 45), 2),
                                "transcription_ur": VOICE_UR[i], "translation_en": VOICE_EN[i],
                                "status": "success"})
            elif r < 0.53:
                events.append((t, f"{name}: <Media omitted>"))
            elif r < 0.55:
                key = t.strftime("%Y%m%d")
                wa_seq[key] = wa_seq.get(key, 0) + 1
                events.append((t, f"{name}: IMG-{key}-WA{wa_seq[key]:04d}.jpg (file attached)"))
            elif r < 0.56:
                events.append((t, f"{name}: This message was deleted"))
            elif r < 0.78:
                events.append((t, f"{name}: {rng.choice(ENGLISH).format(v=rng.choice(VILLAGES))}"))
            elif r < 0.95:
                events.append((t, f"{name}: {rng.choice(ROMAN_URDU).format(v=rng.choice(VILLAGES))}"))
            else:
                events.append((t, f"{name}: {rng.choice(URDU_SCRIPT)}"))
        day += timedelta(days=1)

    # a few fixed messages exercising the anonymiser (fake numbers, mentions, in-body names)
    events += [
        (datetime(2024, 11, 20, 11, 5), f"{SENDERS['sup']}: Please call the control room on 0300 0000000 if the radio fails"),
        (datetime(2024, 12, 5, 16, 40), f"{SENDERS['coord']}: @⁨{SENDERS['emp1']}⁩ please send the report"),
        (datetime(2025, 1, 20, 9, 15), f"{SENDERS['sup']}: Heron and Plover, well done on the night call"),
        (datetime(2025, 1, 28, 12, 0), f"{SENDERS['sup']}: Dr Sparrow from the district office will visit on Monday"),
        (datetime(2025, 2, 18, 21, 50), f"{SENDERS['drv']}: Gari ka tyre puncture, spare laga diya hai\nab theek hai"),
    ]
    events.sort(key=lambda e: e[0])
    lines = [f"{fmt_ts(t)} - {body}" for t, body in events]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "synthetic_whatsapp_export.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(vn_rows).to_csv(OUT_DIR / "synthetic_voice_notes.csv", index=False)
    print(f"[repro_00] {len(events)} lines/messages, {case_no - 1} case reports, {len(vn_rows)} voice notes")


if __name__ == "__main__":
    main()
