#!/usr/bin/env python3
"""M51 Aufgabe 6: OSS-Samples aus VM-Metadaten in samples.json uebernehmen.

Liest /root/voice-test.eu/audio/oss/{model}/_meta.json (von der GPU-VM
generiert) und traegt die Samples kumulativ in data/samples.json ein.
Wandelt die VM-Pfade (/root/out/audio/oss/...) in repo-relative Pfade
(audio/oss/...) um.

Idempotent: existierende Sample-IDs werden übersprungen.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_OSS = ROOT / "audio" / "oss"
SAMPLES_JSON = ROOT / "data" / "samples.json"
TEXTS_JSON = ROOT / "data" / "texts.json"

# Sprache -> Sprachname Mapping fuer Anzeige im Frontend
LANG_NAMES = {
    "de": "Deutsch", "en": "Englisch", "fr": "Französisch",
    "es": "Spanisch", "it": "Italienisch", "pt": "Portugiesisch",
    "nl": "Niederländisch", "ru": "Russisch", "pl": "Polnisch",
    "tr": "Türkisch", "ja": "Japanisch", "zh": "Chinesisch",
    "ko": "Koreanisch", "hi": "Hindi", "bg": "Bulgarisch",
    "cs": "Tschechisch", "da": "Dänisch", "el": "Griechisch",
    "et": "Estnisch", "fi": "Finnisch", "ga": "Irisch",
    "hr": "Kroatisch", "hu": "Ungarisch", "lt": "Litauisch",
    "lv": "Lettisch", "mt": "Maltesisch", "no": "Norwegisch",
    "ro": "Rumänisch", "sk": "Slowakisch", "sl": "Slowenisch",
    "sv": "Schwedisch",
}


def provider_display(model_key: str) -> str:
    return {
        "kokoro": "Kokoro-82M (OSS)",
        "bark": "Bark (OSS)",
        "piper": "Piper (OSS)",
    }.get(model_key, model_key)


def provider_url(model_key: str) -> str:
    return {
        "kokoro": "https://huggingface.co/hexgrad/Kokoro-82M",
        "bark": "https://github.com/suno-ai/bark",
        "piper": "https://github.com/rhasspy/piper",
    }.get(model_key, "")


def main() -> int:
    if not SAMPLES_JSON.exists():
        print("FEHLER: samples.json nicht gefunden", flush=True)
        return 1
    samples_data = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    existing = samples_data.get("samples", [])
    existing_ids = {s["id"] for s in existing}
    print(f"Bestand: {len(existing)} Samples", flush=True)

    if not AUDIO_OSS.exists():
        print(f"FEHLER: {AUDIO_OSS} existiert nicht", flush=True)
        return 1

    new_samples: list[dict] = []
    for model_dir in sorted(AUDIO_OSS.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "_meta.json"
        if not meta_path.exists():
            print(f"  SKIP {model_dir.name}: kein _meta.json", flush=True)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = model_dir.name  # kokoro, bark, piper, ...
        meta_samples = meta.get("samples", [])
        print(f"  {model_name}: {len(meta_samples)} Samples in _meta.json",
              flush=True)
        for s in meta_samples:
            # VM-Pfad -> repo-Pfad
            audio_path = s["audio_path"]
            if "/root/out/audio/oss/" in audio_path:
                audio_path = "audio/oss/" + audio_path.split("/root/out/audio/oss/", 1)[1]
            elif audio_path.startswith("/root/"):
                # unbekannter Pfad, skip
                continue
            sid = f"oss-{model_name}-{s['voice_id']}-{s['language']}-{s['variant']}"
            if sid in existing_ids:
                continue
            # Datei physisch vorhanden?
            full = ROOT / audio_path
            if not full.exists() or full.stat().st_size < 1024:
                continue
            new_samples.append({
                "id": sid,
                "provider": model_name,
                "provider_display": provider_display(model_name),
                "voice_id": s["voice_id"],
                "voice_name": s["voice_name"],
                "language": s["language"],
                "language_name": LANG_NAMES.get(s["language"], s["language"]),
                "variant": s["variant"],
                "gender": s.get("gender", "unknown"),
                "model_type": s.get("model_type", model_name),
                "model_size_mb": s.get("model_size_mb"),
                "audio_path": audio_path,
                "provider_url": provider_url(model_name),
                "license": "oss",
                "generation_time_ms": s.get("generation_time_ms"),
                "sample_rate": s.get("sample_rate", 22050),
                "oss_model": meta.get("model", model_name),
                "oss_framework": meta.get("framework", model_name),
            })

    if not new_samples:
        print("Keine neuen Samples zum Hinzufuegen.", flush=True)
        return 0

    all_samples = existing + new_samples
    samples_data["_meta"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total": len(all_samples),
    }
    samples_data["samples"] = all_samples
    SAMPLES_JSON.write_text(
        json.dumps(samples_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n+{len(new_samples)} Samples hinzugefuegt.", flush=True)
    print(f"samples.json: {len(all_samples)} Samples total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
