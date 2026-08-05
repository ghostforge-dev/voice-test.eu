#!/usr/bin/env python3
"""M51 Aufgabe 3: Google Cloud TTS Samples via Vault-Service-Account.

Liest die SA-JSON aus dem Dev-Brain Vault (google_agentplatform),
holt OAuth2-Token via JWT-Exchange, ruft Cloud TTS API auf:
  GET  /v1/voices           -> alle Stimmen listen
  POST /v1/text:synthesize  -> MP3 generieren (base64 AudioContent)

Endpoint: https://texttospeech.googleapis.com/v1/
Output:
  audio/google/{voice_id}/{lang}_{variant}.mp3
  data/samples.json (append-safe)

Token Refresh autom. nach 50 Min (Token ist 1h gueltig).
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

ROOT = Path(__file__).resolve().parent.parent
DEVBRAIN_BACKEND = Path("/root/dev-brain/backend")
sys.path.insert(0, str(DEVBRAIN_BACKEND))
import secrets_vault  # noqa: E402

AUDIO_DIR = ROOT / "audio"
SAMPLES_JSON = ROOT / "data" / "samples.json"
TEXTS_JSON = ROOT / "data" / "texts.json"

VARIANTS = ["v1_simple", "v2_medium", "v3_emotional"]
TTS_BASE = "https://texttospeech.googleapis.com/v1"
MAX_RETRIES = 2
CONCURRENCY = 5

# Nur EU-Sprachen + EN (s. texts.json)
EU_LANGS_FULL = {
    "bg-BG": "bg", "cs-CZ": "cs", "da-DK": "da", "de-DE": "de",
    "el-GR": "el", "en-GB": "en", "en-US": "en",
    "es-ES": "es", "et-EE": "et", "fi-FI": "fi",
    "fr-CA": "fr", "fr-FR": "fr",
    "ga-IE": "ga", "hr-HR": "hr", "hu-HU": "hu",
    "it-IT": "it", "lt-LT": "lt", "lv-LV": "lv",
    "mt-MT": "mt", "nb-NO": "no", "nl-BE": "nl", "nl-NL": "nl",
    "pl-PL": "pl", "pt-PT": "pt", "pt-BR": "pt",
    "ro-RO": "ro", "sk-SK": "sk", "sl-SI": "sl", "sv-SE": "sv",
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


async def get_access_token(client: httpx.AsyncClient, sa: dict) -> str:
    """JWT-basierter OAuth2-Token-Exchange (RS256)."""
    now = int(time.time())
    payload = {
        "iss": sa["client_email"],
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": sa["token_uri"],
        "iat": now,
        "exp": now + 3600,
    }
    header = {"alg": "RS256", "typ": "JWT", "kid": sa.get("private_key_id")}

    sign_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )

    private_key = serialization.load_pem_private_key(
        sa["private_key"].encode(), password=None,
    )
    signature = private_key.sign(
        sign_input.encode("ascii"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    jwt_token = sign_input + "." + _b64url(signature)

    r = await client.post(
        sa["token_uri"],
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OAuth-Exchange HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["access_token"]


async def list_voices(client: httpx.AsyncClient, token: str) -> list[dict]:
    """GET /v1/voices -> Liste aller Cloud-TTS-Stimmen, gefiltert auf EU."""
    r = await client.get(
        f"{TTS_BASE}/voices",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"list_voices HTTP {r.status_code}: {r.text[:300]}")
    all_voices = r.json().get("voices", [])
    # Gruppiere nach voice_id (gleicher Name kann mehrere Sprachen haben)
    grouped: dict[str, dict] = {}
    skipped_unsupported = 0
    for v in all_voices:
        # Nur Standard/Wavenet/Neural2/Studio (Chirp+Journey skipped)
        if _model_type(v["name"]) in UNSUPPORTED_TYPES:
            skipped_unsupported += 1
            continue
        for lc in v.get("languageCodes", []):
            short = EU_LANGS_FULL.get(lc)
            if not short:
                continue
            vid = v["name"]
            if vid not in grouped:
                grouped[vid] = {
                    "id": vid,
                    "name": vid,
                    "gender": v.get("ssmlGender", "SSML_VOICE_GENDER_UNSPECIFIED"),
                    "language_codes": [],
                    "short_langs": [],
                }
            if short not in grouped[vid]["short_langs"]:
                grouped[vid]["short_langs"].append(short)
                grouped[vid]["language_codes"].append(lc)
    if skipped_unsupported:
        print(f"  ({skipped_unsupported} Chirp/Journey-Stimmen übersprungen)", flush=True)
    return list(grouped.values())


def _model_type(voice_id: str) -> str:
    name_lower = voice_id.lower()
    if "wavenet" in name_lower:
        return "wavenet"
    if "neural2" in name_lower:
        return "neural2"
    if "studio" in name_lower:
        return "studio"
    if "chirp" in name_lower:
        return "chirp"
    if "journey" in name_lower:
        return "journey"
    # Chirp1-Stimmen haben Eigennamen ("Achernar", "Achird") ohne BCP47-Prefix.
    # Standard-Wavenet/Neural2/Studio haben Pattern wie "de-DE-Standard-A"
    if not _has_bcp47_prefix(voice_id):
        return "chirp"
    return "standard"


def _has_bcp47_prefix(name: str) -> bool:
    """True wenn der Name mit BCP47-Sprachcode beginnt (xx-XX oder xx-XXX)."""
    parts = name.split("-")
    if len(parts) < 2:
        return False
    # Erster Teil: 2-3 Buchstaben (Sprachcode)
    if not (2 <= len(parts[0]) <= 3 and parts[0].isalpha()):
        return False
    # Zweiter Teil: 2-3 Buchstaben (Region oder Type-Kennung)
    if not (2 <= len(parts[1]) <= 4 and parts[1].isalpha()):
        return False
    return True


# Stimmen, die mit normalem synthesize()-Aufruf nicht funktionieren
# (brauchen spezielle Parameter / sind nur via Vertex AI Chirp verfuegbar)
UNSUPPORTED_TYPES = {"chirp", "journey"}


def _gender_normalize(g: str) -> str:
    g = (g or "").replace("SSML_VOICE_GENDER_", "").lower()
    return g if g in ("male", "female", "neutral") else "unknown"


async def synth_one(
    client: httpx.AsyncClient,
    token_holder: dict,
    sa: dict,
    voice: dict,
    lang_short: str,
    variant: str,
    text: str,
    out_path: Path,
) -> int:
    """POST /v1/text:synthesize -> MP3 speichern."""
    # Passendes BCP-47 languageCode fuer die Sprache finden
    matching_lc = None
    for lc in voice["language_codes"]:
        if EU_LANGS_FULL.get(lc) == lang_short:
            matching_lc = lc
            break
    if not matching_lc:
        raise RuntimeError(
            f"kein languageCode fuer {lang_short} an voice {voice['id']} "
            f"(hat {voice['language_codes']})"
        )

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": matching_lc,
            "name": voice["id"],
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "sampleRateHertz": 24000
        },
    }
    t0 = time.monotonic()
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            r = await client.post(
                f"{TTS_BASE}/text:synthesize",
                headers={
                    "Authorization": f"Bearer {token_holder['token']}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            if r.status_code == 401:
                # Token expired -> refresh
                token_holder["token"] = await get_access_token(client, sa)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"synthesize HTTP {r.status_code}: {r.text[:200]}")
            audio_b64 = r.json().get("audioContent", "")
            audio_bytes = base64.b64decode(audio_b64)
            if len(audio_bytes) < 256:
                raise RuntimeError("leere Audio-Antwort")
            out_path.write_bytes(audio_bytes)
            return int((time.monotonic() - t0) * 1000)
        except httpx.HTTPError:
            if attempt > MAX_RETRIES:
                raise
            await asyncio.sleep(1.5 * attempt)
    return int((time.monotonic() - t0) * 1000)


def _tag_mp3(path: Path, provider: str, voice_name: str) -> None:
    try:
        from mutagen.mp3 import MP3  # type: ignore
        from mutagen.id3 import ID3, TPE1, COMM, TCOP  # type: ignore
        try:
            audio = MP3(str(path))
            if audio.tags is None:
                audio.add_tags()
            tag = audio.tags
        except Exception:
            tag = ID3()
        tag.add(TPE1(encoding=3, text=["Ghostforge"]))
        tag.add(COMM(encoding=3, lang="eng", desc="Source",
                     text=[f"Generated by voice-test.dev -- {provider} {voice_name}"]))
        tag.add(TCOP(encoding=3, text=["CC BY 4.0 -- voice-test.dev"]))
        if isinstance(tag, ID3):
            tag.save(str(path))
        else:
            audio.save()
    except Exception as e:
        print(f"    [tag warn] {path.name}: {e}", flush=True)


async def main() -> int:
    sa_raw = await secrets_vault.get_secret("google_agentplatform", "credentials_json")
    sa = json.loads(sa_raw)
    print(f"SA: {sa['client_email']}", flush=True)

    texts = json.loads(TEXTS_JSON.read_text(encoding="utf-8"))
    available_langs = set(texts["texts"]["v1_simple"].keys())

    existing = []
    if SAMPLES_JSON.exists():
        try:
            existing = json.loads(
                SAMPLES_JSON.read_text(encoding="utf-8")
            ).get("samples", [])
        except Exception:
            pass
    existing_ids = {s["id"] for s in existing}

    async with httpx.AsyncClient() as client:
        token_holder = {"token": await get_access_token(client, sa)}
        print("OAuth-Token erhalten", flush=True)

        voices = await list_voices(client, token_holder["token"])
        print(f"{len(voices)} Stimmen mit EU-Sprachen gefunden", flush=True)

        # Job-Liste
        plan_jobs: list[tuple] = []
        for v in voices:
            for lang in v["short_langs"]:
                if lang not in available_langs:
                    continue
                for variant in VARIANTS:
                    sid = f"google-{v['id']}-{lang}-{variant}"
                    safe_voice = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in v["id"]
                    )
                    out_path = AUDIO_DIR / "google" / safe_voice / f"{lang}_{variant}.mp3"
                    if sid in existing_ids:
                        continue
                    if out_path.exists() and out_path.stat().st_size > 1024:
                        continue
                    text = texts["texts"][variant].get(lang)
                    if not text:
                        continue
                    plan_jobs.append((v, lang, variant, text, sid, out_path))

        print(f"Plan: {len(plan_jobs)} Samples zu generieren", flush=True)
        if not plan_jobs:
            print("  Nichts zu tun.", flush=True)
            return 0

        sem = asyncio.Semaphore(CONCURRENCY)
        counters = {"ok": 0, "err": 0}
        errors: list[str] = []
        new_samples: list[dict] = []
        done = 0

        async def worker(v, lang, variant, text, sid, out_path):
            nonlocal done
            async with sem:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    ms = await synth_one(
                        client, token_holder, sa, v, lang, variant, text, out_path,
                    )
                    safe_voice = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in v["id"]
                    )
                    _tag_mp3(out_path, "google", v["id"])
                    new_samples.append({
                        "id": sid,
                        "provider": "google",
                        "provider_display": "Google Cloud TTS",
                        "voice_id": v["id"],
                        "voice_name": v["id"],
                        "language": lang,
                        "language_name": next(
                            (l["name"] for l in texts.get("languages", [])
                             if l.get("code") == lang), lang),
                        "variant": variant,
                        "gender": _gender_normalize(v["gender"]),
                        "model_type": _model_type(v["id"]),
                        "model_size_mb": None,
                        "audio_path": str(out_path.relative_to(ROOT)),
                        "provider_url": "https://cloud.google.com/text-to-speech",
                        "license": "proprietary",
                        "generation_time_ms": ms,
                        "sample_rate": 24000,
                    })
                    counters["ok"] += 1
                except Exception as e:
                    counters["err"] += 1
                    errors.append(f"google/{v['id']}/{lang}_{variant}: {e}")
                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                done += 1
                if done % 50 == 0:
                    print(f"  Progress: {done}/{len(plan_jobs)} "
                          f"(ok={counters['ok']}, err={counters['err']})", flush=True)

        tasks = [worker(*job) for job in plan_jobs]
        for i in range(0, len(tasks), 100):
            await asyncio.gather(*tasks[i:i + 100])

    # Sync auf Disk vorhandener aber nicht in samples.json erfasster Samples
    for v in voices:
        safe_voice = "".join(c if c.isalnum() or c in "-_" else "_" for c in v["id"])
        for lang in v["short_langs"]:
            if lang not in available_langs:
                continue
            for variant in VARIANTS:
                sid = f"google-{v['id']}-{lang}-{variant}"
                if sid in existing_ids or any(s["id"] == sid for s in new_samples):
                    continue
                out_path = AUDIO_DIR / "google" / safe_voice / f"{lang}_{variant}.mp3"
                if out_path.exists() and out_path.stat().st_size > 1024:
                    new_samples.append({
                        "id": sid,
                        "provider": "google",
                        "provider_display": "Google Cloud TTS",
                        "voice_id": v["id"],
                        "voice_name": v["id"],
                        "language": lang,
                        "language_name": next(
                            (l["name"] for l in texts.get("languages", [])
                             if l.get("code") == lang), lang),
                        "variant": variant,
                        "gender": _gender_normalize(v["gender"]),
                        "model_type": _model_type(v["id"]),
                        "model_size_mb": None,
                        "audio_path": str(out_path.relative_to(ROOT)),
                        "provider_url": "https://cloud.google.com/text-to-speech",
                        "license": "proprietary",
                        "generation_time_ms": None,
                        "sample_rate": 24000,
                    })

    all_samples = existing + new_samples
    SAMPLES_JSON.write_text(
        json.dumps({"_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total": len(all_samples),
        }, "samples": all_samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFertig: ok={counters['ok']}, err={counters['err']}", flush=True)
    print(f"samples.json: {len(all_samples)} Samples total", flush=True)
    if errors:
        print(f"Fehler ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
