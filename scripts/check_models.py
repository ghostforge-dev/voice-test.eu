#!/usr/bin/env python3
"""M51 S6: Woechentlicher Model-Check fuer HuggingFace-TTS-Modelle.

Laeuft in GitHub Actions (Montag 03:00 UTC) oder manuell. Macht:

  1. HuggingFace API: alle text-to-speech-Modelle paginieren
  2. Filter: downloads >= 500, OSS-Lizenz, >= 1 EU-Sprache
  3. Mit bestehender data/models.json abgleichen
  4. Neue Modelle mit samples_generated=false hinzufuegen
  5. Tote Links als deprecated markieren (HTTP HEAD)
  6. models.json aktualisieren (Commit passiert via Action)

Lokaler Aufruf:  python scripts/check_models.py
In GitHub Actions: workflow_dispatch oder schedule.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = ROOT / "data" / "models.json"
HF_API = "https://huggingface.co/api/models"

EU_LANG_CODES = {
    "de", "en", "fr", "es", "it", "pt", "pl", "nl", "cs", "el",
    "hu", "ro", "sv", "da", "fi", "sk", "bg", "hr", "sl", "et",
    "lv", "lt", "ga", "mt", "no",
}

OSS_LICENSES = {
    "apache-2.0", "mit", "mpl-2.0", "bsd-3-clause", "bsd-2-clause",
    "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc0-1.0",
    "cpml", "openrail++", "llama2", "llama3", "llama3.1", "gemma",
    "afl-3.0", "isc", "unlicense", "osl-3.0", "cc-by-nc-4.0",
}

MIN_DOWNLOADS = 500


def fetch_all_tts_models() -> list[dict]:
    """Paginiert alle TTS-Modelle von HF."""
    models = []
    params = {"pipeline_tag": "text-to-speech", "full": "true", "limit": 100}
    with httpx.Client(timeout=30) as c:
        skip = 0
        while True:
            params["skip"] = skip
            r = c.get(HF_API, params=params)
            if r.status_code != 200:
                print(f"HF API HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
                break
            batch = r.json()
            if not batch:
                break
            models.extend(batch)
            if len(batch) < 100:
                break
            skip += 100
            time.sleep(0.5)  # HF rate-limit schonend
    return models


def extract_license(tags: list[str], explicit: str | None) -> str | None:
    if explicit:
        return explicit.lower()
    for tag in tags:
        t = tag.lower()
        if t.startswith("license:"):
            return t.split(":", 1)[1].strip()
        if t in OSS_LICENSES:
            return t
    return None


def extract_eu_langs(tags: list[str], model_id: str) -> list[str]:
    found = set()
    for tag in tags:
        t = tag.strip().lower()
        if t in EU_LANG_CODES:
            found.add(t)
        if t.startswith("language:") and t.split(":")[1] in EU_LANG_CODES:
            found.add(t.split(":")[1])
    m = re.search(r"\b(de|en|fr|es|it|pt|pl|nl|cs|el|hu|ro|sv|da|fi|sk|bg|hr|sl|et|lv|lt|ga|mt|no)\b",
                  model_id.lower())
    if m:
        found.add(m.group(1))
    return sorted(found)


def detect_framework(model_id: str, tags: list[str]) -> str:
    blob = f"{model_id.lower()} {' '.join(tags).lower()}"
    for pattern, fw in [
        (r"kokoro", "kokoro"), (r"xtts", "xtts"), (r"piper", "piper"),
        (r"\bbark\b", "bark"), (r"speecht5|speech-t5", "speecht5"),
        (r"tortoise", "tortoise"), (r"parler", "parlertts"),
        (r"openvoice", "openvoice"), (r"fish-speech|fish_tts", "fish-speech"),
        (r"melo", "melo"), (r"styletts", "styletts"),
        (r"vits", "vits"), (r"tacotron", "tacotron"),
    ]:
        if re.search(pattern, blob):
            return fw
    return "unknown"


def main() -> int:
    print("S6: Model-Check (HuggingFace)", flush=True)
    existing_data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    existing = {m["id"]: m for m in existing_data.get("models", [])}
    print(f"  Bestehende Modelle: {len(existing)}", flush=True)

    all_models = fetch_all_tts_models()
    print(f"  HF-Modelle gesamt: {len(all_models)}", flush=True)

    new_count = 0
    updated_count = 0
    for m in all_models:
        mid = m.get("id")
        if not mid:
            continue
        downloads = m.get("downloads") or 0
        if downloads < MIN_DOWNLOADS:
            continue
        tags = m.get("tags") or []
        license_ = extract_license(tags, m.get("license"))
        if not license_ or license_ not in OSS_LICENSES:
            continue
        eu_langs = extract_eu_langs(tags, mid)
        if not eu_langs:
            continue

        if mid in existing:
            # Bestehendes Modell updaten (downloads, last_checked)
            existing[mid]["downloads"] = downloads
            existing[mid]["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            updated_count += 1
        else:
            # Neues Modell
            framework = detect_framework(mid, tags)
            new_model = {
                "id": mid,
                "name": mid.split("/")[-1].split("-")[0],
                "source": "huggingface",
                "license": license_,
                "license_oss": True,
                "downloads": downloads,
                "size_mb": None,
                "languages": eu_langs,
                "eu_languages": eu_langs,
                "model_url": f"https://huggingface.co/{mid}",
                "framework": framework,
                "last_checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "active",
                "samples_generated": False,
                "new_at": time.strftime("%Y-%m-%d"),
            }
            existing[mid] = new_model
            new_count += 1
            print(f"    NEU: {mid} ({downloads} DL, {framework})", flush=True)

    # Als Liste sortiert nach Downloads
    all_list = sorted(existing.values(), key=lambda x: x.get("downloads") or 0, reverse=True)

    # Link-Check (HEAD-Request auf model_url) fuer top 60
    print("  Link-Check (Top 60)...", flush=True)
    broken = 0
    with httpx.Client(timeout=10) as c:
        for m in all_list[:60]:
            try:
                r = c.head(m["model_url"], follow_redirects=True)
                if r.status_code != 200:
                    m["status"] = "deprecated"
                    m["deprecated_reason"] = f"HTTP {r.status_code}"
                    broken += 1
            except Exception as e:  # noqa: BLE001
                m["status"] = "deprecated"
                m["deprecated_reason"] = str(e)[:100]
                broken += 1

    out = {
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total": len(all_list),
            "source": "huggingface",
            "filter": f"downloads>={MIN_DOWNLOADS}, >=1 EU language, OSS license",
            "new_this_run": new_count,
            "broken_links": broken,
        },
        "models": all_list,
    }
    MODELS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {len(all_list)} Modelle, {new_count} neu, {updated_count} aktualisiert, {broken} kaputte Links", flush=True)
    print(f"Geschrieben: {MODELS_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
