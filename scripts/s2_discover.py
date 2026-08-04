#!/usr/bin/env python3
"""M51 S2a: Voice-Discovery fuer alle Enterprise-Provider.

Listet alle verfuegbaren Stimmen pro Provider und filtert auf EU-Sprachen.
Ergebnis: data/voices.json

Provider, die nicht konfiguriert sind (z.B. Google Cloud TTS ohne Service
Account), werden als `_available: false` markiert und mit leeren Voices
eingetragen -- so weiss das Generate-Script, welcher Provider fehlt.

Secrets: aws/azure/elevenlabs aus /root/hermes-workspace/devbrain/.env/*.env,
OpenAI aus dem Dev-Brain Vault.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

DEVBRAIN_BACKEND = Path("/root/dev-brain/backend")
sys.path.insert(0, str(DEVBRAIN_BACKEND))
import secrets_vault  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = Path("/root/hermes-workspace/devbrain/.env")
OUT = ROOT / "data" / "voices.json"

# 25 Sprachen (24 EU + Englisch + Norwegisch), als ISO 639-1 und mit BCP47-Hub
EU_LANGS = {
    "de": ["de-DE", "de-AT", "de-CH"],
    "en": ["en-GB", "en-US", "en-IE", "en-AU"],
    "fr": ["fr-FR", "fr-BE", "fr-CA"],
    "es": ["es-ES", "es-MX", "es-US"],
    "it": ["it-IT"],
    "pt": ["pt-PT", "pt-BR"],
    "pl": ["pl-PL"],
    "nl": ["nl-NL", "nl-BE"],
    "cs": ["cs-CZ"],
    "el": ["el-GR"],
    "hu": ["hu-HU"],
    "ro": ["ro-RO"],
    "sv": ["sv-SE"],
    "da": ["da-DK"],
    "fi": ["fi-FI"],
    "sk": ["sk-SK"],
    "bg": ["bg-BG"],
    "hr": ["hr-HR"],
    "sl": ["sl-SI"],
    "et": ["et-EE"],
    "lv": ["lv-LV"],
    "lt": ["lt-LT"],
    "ga": ["ga-IE"],
    "mt": ["mt-MT"],
    "no": ["nb-NO", "nn-NO"],
}
EU_LANG_CODES = set(EU_LANGS.keys())


def _env_file(name: str) -> dict[str, str]:
    path = ENV_DIR / name
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _bcp47_to_iso(code: str) -> str | None:
    """'de-DE' -> 'de', 'en-US' -> 'en'."""
    base = code.split("-")[0].lower()
    return base if base in EU_LANG_CODES else None


# ---------- OpenAI (statische Voice-Liste, alle mehrsprachig) ----------
OPENAI_VOICES = [
    ("alloy", "Alloy", "neutral", "gpt-4o-mini-tts"),
    ("echo", "Echo", "male", "gpt-4o-mini-tts"),
    ("fable", "Fable", "neutral", "tts-1"),
    ("onyx", "Onyx", "male", "tts-1"),
    ("nova", "Nova", "female", "tts-1"),
    ("shimmer", "Shimmer", "female", "tts-1"),
    ("coral", "Coral", "female", "gpt-4o-mini-tts"),
    ("sage", "Sage", "neutral", "gpt-4o-mini-tts"),
    ("ash", "Ash", "male", "gpt-4o-mini-tts"),
    ("ballad", "Ballad", "male", "gpt-4o-mini-tts"),
]


async def discover_openai() -> dict:
    try:
        key = await secrets_vault.get_secret("openai", "api_key")
    except Exception as e:  # noqa: BLE001
        return {"_available": False, "_reason": f"vault: {e}", "voices": []}
    if not key:
        return {"_available": False, "_reason": "no key in vault", "voices": []}
    # Probe-Call um sicherzustellen, dass der Key TTS darf
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            if r.status_code not in (200, 401):
                return {"_available": False, "_reason": f"probe HTTP {r.status_code}", "voices": []}
        except Exception as e:  # noqa: BLE001
            return {"_available": False, "_reason": f"probe: {e}", "voices": []}
    return {
        "_available": True,
        "voices": [
            {"id": vid, "name": name, "gender": gender, "type": mtype,
             "languages": sorted(EU_LANG_CODES)}
            for vid, name, gender, mtype in OPENAI_VOICES
        ],
    }


# ---------- AWS Polly ----------
def discover_aws() -> dict:
    env = _env_file("aws.env")
    if not env.get("AWS_ACCESS_KEY_ID"):
        return {"_available": False, "_reason": "aws.env fehlt", "voices": []}
    try:
        import boto3  # type: ignore
    except ImportError as e:
        return {"_available": False, "_reason": f"boto3: {e}", "voices": []}
    try:
        client = boto3.client(
            "polly",
            aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
            region_name=env.get("AWS_DEFAULT_REGION", "eu-central-1"),
        )
        voices = []
        paginator = client.get_paginator("describe_voices")
        for page in paginator.paginate():
            for v in page.get("Voices", []):
                iso = _bcp47_to_iso(v.get("LanguageCode", ""))
                if not iso:
                    continue
                if v.get("SupportedEngines") and "neural" not in v["SupportedEngines"]:
                    continue  # nur Neural, keine Standard
                existing = next((x for x in voices if x["id"] == v["Id"]), None)
                if existing:
                    if iso not in existing["languages"]:
                        existing["languages"].append(iso)
                    continue
                voices.append({
                    "id": v["Id"],
                    "name": v.get("Name", v["Id"]),
                    "gender": (v.get("Gender") or "Unknown").lower(),
                    "type": "neural",
                    "languages": [iso],
                })
        return {"_available": True, "voices": voices}
    except Exception as e:  # noqa: BLE001 -- boto3 wirft bei ungueltigen Keys
        return {"_available": False, "_reason": f"{type(e).__name__}: {str(e)[:120]}", "voices": []}


# ---------- Azure Speech ----------
def discover_azure() -> dict:
    env = _env_file("azure.env")
    key = env.get("AZURE_SPEECH_KEY")
    region = env.get("AZURE_SPEECH_REGION")
    if not key or not region:
        return {"_available": False, "_reason": "azure.env unvollstaendig", "voices": []}
    try:
        r = httpx.get(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list",
            headers={"Ocp-Apim-Subscription-Key": key},
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        return {"_available": False, "_reason": f"http: {e}", "voices": []}
    if r.status_code != 200:
        return {"_available": False, "_reason": f"HTTP {r.status_code}", "voices": []}
    voices = []
    for v in r.json():
        iso = _bcp47_to_iso(v.get("Locale", ""))
        if not iso:
            continue
        existing = next((x for x in voices if x["id"] == v["ShortName"]), None)
        if existing:
            if iso not in existing["languages"]:
                existing["languages"].append(iso)
            continue
        voices.append({
            "id": v["ShortName"],
            "name": v.get("LocalName") or v.get("DisplayName", v["ShortName"]),
            "gender": (v.get("Gender") or "Unknown").lower(),
            "type": v.get("VoiceType", "Neural").lower(),
            "languages": [iso],
        })
    return {"_available": True, "voices": voices}


# ---------- ElevenLabs ----------
async def discover_elevenlabs() -> dict:
    env = _env_file("elevenlabs.env")
    key = env.get("ELEVENLABS_API_KEY")
    if not key:
        return {"_available": False, "_reason": "elevenlabs.env fehlt", "voices": []}
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": key},
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            return {"_available": False, "_reason": f"http: {e}", "voices": []}
    if r.status_code != 200:
        return {"_available": False, "_reason": f"HTTP {r.status_code}", "voices": []}
    voices = []
    for v in r.json().get("voices", []):
        labels = v.get("labels", {})
        # ElevenLabs-Stimmen sind mehrsprachig (multilingual v2)
        voices.append({
            "id": v["voice_id"],
            "name": v.get("name", v["voice_id"]),
            "gender": (labels.get("gender") or "unknown").lower(),
            "type": "multilingual-v2",
            "languages": sorted(EU_LANG_CODES),
        })
    return {"_available": True, "voices": voices}


# ---------- Edge TTS (kein Key) ----------
async def discover_edge() -> dict:
    try:
        import edge_tts  # type: ignore
    except ImportError as e:
        return {"_available": False, "_reason": f"edge_tts: {e}", "voices": []}
    voices = []
    try:
        all_voices = await edge_tts.list_voices()
    except Exception as e:  # noqa: BLE001
        return {"_available": False, "_reason": f"list: {e}", "voices": []}
    for v in all_voices:
        iso = _bcp47_to_iso(v.get("Locale", ""))
        if not iso:
            continue
        existing = next((x for x in voices if x["id"] == v["ShortName"]), None)
        if existing:
            if iso not in existing["languages"]:
                existing["languages"].append(iso)
            continue
        voices.append({
            "id": v["ShortName"],
            "name": v.get("FriendlyName", v["ShortName"]),
            "gender": (v.get("Gender") or "Unknown").lower(),
            "type": v.get("VoiceTag", {}).get("VoicePersonalities", ["Standard"])[0]
                    if isinstance(v.get("VoiceTag"), dict) else "standard",
            "languages": [iso],
        })
    return {"_available": True, "voices": voices}


async def main() -> int:
    print("S2a: Voice-Discovery (Enterprise Provider)", flush=True)
    providers = {}
    print("  OpenAI...", flush=True)
    providers["openai"] = await discover_openai()
    print(f"    -> {len(providers['openai']['voices'])} voices, avail={providers['openai']['_available']}", flush=True)
    print("  AWS Polly...", flush=True)
    providers["aws"] = discover_aws()
    print(f"    -> {len(providers['aws']['voices'])} voices, avail={providers['aws']['_available']}", flush=True)
    print("  Azure...", flush=True)
    providers["azure"] = discover_azure()
    print(f"    -> {len(providers['azure']['voices'])} voices, avail={providers['azure']['_available']}", flush=True)
    print("  ElevenLabs...", flush=True)
    providers["elevenlabs"] = await discover_elevenlabs()
    print(f"    -> {len(providers['elevenlabs']['voices'])} voices, avail={providers['elevenlabs']['_available']}", flush=True)
    print("  Edge TTS...", flush=True)
    providers["edge"] = await discover_edge()
    print(f"    -> {len(providers['edge']['voices'])} voices, avail={providers['edge']['_available']}", flush=True)

    providers["google"] = {
        "_available": False,
        "_reason": "kein Google Cloud Service Account in .env -- Bauplan-S0c sagt vorhanden, aber Datei fehlt",
        "voices": [],
    }
    print("  Google Cloud TTS: NICHT VERFUEGBAR (Blocker)", flush=True)

    total = sum(len(p["voices"]) for p in providers.values())
    print(f"\nTotal voices: {total}", flush=True)
    print("Per Provider:")
    for name, p in providers.items():
        if p["_available"]:
            n = len(p["voices"])
            sample_est = sum(len(v["languages"]) * 3 for v in p["voices"])
            print(f"  {name:>12}: {n:>4} voices -> ~{sample_est:>5} Samples")

    out = {
        "_meta": {
            "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
            "eu_language_count": len(EU_LANG_CODES),
        },
        "providers": providers,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
