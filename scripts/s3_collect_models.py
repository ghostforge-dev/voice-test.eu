#!/usr/bin/env python3
"""M51 S3a: Sammelt und filtert Open-Source TTS-Modelle von HuggingFace.

Quellen:
  - /root/hermes-workspace/devbrain/hf-tts-models-raw.json (Vorarbeit, 6850 Modelle)

Filter (kumulativ):
  1. downloads >= MIN_DOWNLOADS (Default 500)
  2. Lizenz ist OSS (apache-2.0, mit, mpl-2.0, cpml, cc-by-*, etc.)
  3. Mindestens 1 EU-Sprache erkennbar (aus Tags oder Modell-Id)

Framework-Erkennung anhand der Modell-Id / Tags:
  kokoro, xtts, piper, bark, vits, speecht5, gtts, tortoise, parlertts, openvoice, fish-speech, jazz, melo, StyleTTS

Output: data/models.json mit den wichtigsten ~30-50 Modellen, sortiert nach Downloads.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = Path("/root/hermes-workspace/devbrain/hf-tts-models-raw.json")
OUT = ROOT / "data" / "models.json"

MIN_DOWNLOADS = 500

EU_LANG_CODES = {
    "de", "en", "fr", "es", "it", "pt", "pl", "nl", "cs", "el",
    "hu", "ro", "sv", "da", "fi", "sk", "bg", "hr", "sl", "et",
    "lv", "lt", "ga", "mt", "no",
}
LANG_NAME = {
    "de": "Deutsch", "en": "English", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "nl": "Dutch",
    "cs": "Czech", "el": "Greek", "hu": "Hungarian", "ro": "Romanian",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "sk": "Slovak",
    "bg": "Bulgarian", "hr": "Croatian", "sl": "Slovenian", "et": "Estonian",
    "lv": "Latvian", "lt": "Lithuanian", "ga": "Irish", "mt": "Maltese",
    "no": "Norwegian",
}

# Lizenzen, die wir als OSS akzeptieren (klein geschrieben)
OSS_LICENSES = {
    "apache-2.0", "mit", "mpl-2.0", "bsd-3-clause", "bsd-2-clause",
    "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc0-1.0",
    "cpml", "openrail++", "llama2", "llama3", "llama3.1", "gemma",
    "afl-3.0", "isc", "unlicense", "osl-3.0",
    "cc-by-nc-4.0",  # non-commercial, aber Source-offen
}

# Framework-Erkennung
FRAMEWORK_PATTERNS = [
    (r"kokoro", "kokoro"),
    (r"xtts", "xtts"),
    (r"piper", "piper"),
    (r"\bbark\b", "bark"),
    (r"speecht5|speech-t5", "speecht5"),
    (r"tortoise", "tortoise"),
    (r"parler", "parlertts"),
    (r"openvoice", "openvoice"),
    (r"fish-speech|fish_tts", "fish-speech"),
    (r"jenny-tts|jenny_tts", "jenny"),
    (r"melo", "melo"),
    (r"styletts", "styletts"),
    (r"gtts", "gtts"),
    (r"vits", "vits"),    # generisch, letzter Check
    (r"tacotron", "tacotron"),
    (r"fastspeech", "fastspeech"),
]


def _extract_license(tags: list[str], explicit: str | None) -> str | None:
    if explicit:
        return explicit.lower()
    for tag in tags:
        if tag.lower().startswith("license:"):
            return tag.split(":", 1)[1].strip().lower()
    # Manche Tags sind direkt die Lizenz
    for tag in tags:
        t = tag.lower()
        if t in OSS_LICENSES:
            return t
    return None


def _extract_languages(tags: list[str], model_id: str) -> list[str]:
    found = set()
    # ISO 639-1 Codes in tags
    for tag in tags:
        t = tag.strip().lower()
        if t in EU_LANG_CODES:
            found.add(t)
        # "language:de" Form
        if t.startswith("language:") and t.split(":")[1] in EU_LANG_CODES:
            found.add(t.split(":")[1])
    # Aus Modell-Id: "vits-de" oder "tts_german" etc.
    m = re.search(r"\b(de|en|fr|es|it|pt|pl|nl|cs|el|hu|ro|sv|da|fi|sk|bg|hr|sl|et|lv|lt|ga|mt|no)\b",
                  model_id.lower())
    if m:
        found.add(m.group(1))
    # Ausgeschriebene Sprachen in tags
    name_to_code = {v.lower(): k for k, v in LANG_NAME.items()}
    for tag in tags:
        t = tag.strip().lower()
        if t in name_to_code:
            found.add(name_to_code[t])
    return sorted(found)


def _detect_framework(model_id: str, tags: list[str]) -> str:
    blob = f"{model_id.lower()} {' '.join(tags).lower()}"
    for pattern, fw in FRAMEWORK_PATTERNS:
        if re.search(pattern, blob):
            return fw
    return "unknown"


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1].split("-")[0]


def main() -> int:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    print(f"S3a: {len(raw)} raw HF-Modelle, filtere...", flush=True)

    candidates = []
    for m in raw:
        tags = m.get("tags") or []
        downloads = m.get("downloads") or 0
        if downloads < MIN_DOWNLOADS:
            continue
        # License muss da sein (OSS oder proprietär muss erkennbar)
        license_ = _extract_license(tags, m.get("license"))
        # Sprache muss erkennbar sein (mindestens 1 EU)
        eu_langs = _extract_languages(tags, m["id"])
        if not eu_langs:
            continue
        framework = _detect_framework(m["id"], tags)

        candidates.append({
            "id": m["id"],
            "name": _short_name(m["id"]),
            "source": "huggingface",
            "license": license_ or "unknown",
            "license_oss": (license_ or "") in OSS_LICENSES if license_ else False,
            "downloads": downloads,
            "size_mb": None,  # spaeter via HF API ergaenzen
            "languages": eu_langs,
            "eu_languages": eu_langs,
            "model_url": f"https://huggingface.co/{m['id']}",
            "framework": framework,
            "last_checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "active",
            "samples_generated": False,
        })

    # Sortiere nach Downloads (absteigend)
    candidates.sort(key=lambda x: x["downloads"], reverse=True)
    print(f" Nach Filter: {len(candidates)} Modelle", flush=True)

    # Top-K das wir behalten (Tier 1: OSS + bekanntes Framework)
    top = []
    seen_frameworks = set()
    for c in candidates:
        if c["framework"] != "unknown" and c["license_oss"]:
            top.append(c)
            seen_frameworks.add(c["framework"])
    # Tier 2: Rest mit EU-Sprache und OSS Lizenz
    for c in candidates:
        if c in top:
            continue
        if c["license_oss"]:
            top.append(c)

    # Limit auf 60 Modelle
    top = top[:60]
    print(f" Top-Liste: {len(top)} Modelle", flush=True)
    print(f" Davo OSS-Lizenz: {sum(1 for c in top if c['license_oss'])}")
    print(f" Frameworks: {sorted({c['framework'] for c in top})}")

    out = {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total": len(top),
            "source": "huggingface",
            "filter": f"downloads>={MIN_DOWNLOADS}, >=1 EU language",
        },
        "models": top,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}", flush=True)
    print("\nTop 10:")
    for c in top[:10]:
        print(f"  {c['downloads']:>10}  {c['framework']:>10}  {c['license']:>15}  "
              f"{'/'.join(c['eu_languages'][:5]):<25}  {c['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
