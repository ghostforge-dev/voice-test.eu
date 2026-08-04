#!/usr/bin/env python3
"""M51 S3b: GPU-VM Sample-Generierung fuer Open-Source Modelle.

Laeuft AUF der Scaleway-L4-VM (NICHT lokal). Installiert:
  - piper-tts (rhasspy/piper-voices, CPU oder CUDA)
  - Bark, Kokoro, XTTS-v2 (via Coqui TTS / eigenstaendig)

Input:  ./texts.json (mitgebracht vom Host)
Output: ./out/oss_samples.json + ./out/audio/oss/{framework}/{model}/{lang}_{variant}.wav

Strategie: Pro Framework 1 Modell laden, fuer alle EU-Sprachen Samples machen
die das Modell nativ unterstuetzt, dann entladen. Spart GPU-Speicher.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.cuda.empty_cache()
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

ROOT = Path(__file__).resolve().parent
TEXTS = ROOT / "texts.json"
OUT = ROOT / "out"
AUDIO = OUT / "audio" / "oss"
META = OUT / "oss_samples.json"
AUDIO.mkdir(parents=True, exist_ok=True)

VARIANTS = ["v1_simple", "v2_medium", "v3_emotional"]

FRAMEWORK_DISPLAY = {
    "piper": "Piper (rhasspy)",
    "xtts": "Coqui XTTS-v2",
    "kokoro": "Kokoro-82M",
    "bark": "Suno Bark",
}

# Piper-Voice-Mapping (lang_code -> voice file name on HF)
# Voicenamen via HF-API verifiziert (Aug 2026)
PIPER_VOICES = {
    "de": "de_DE-thorsten-medium",
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
    "it": "it_IT-paola-medium",
    "pt": "pt_BR-faber-medium",        # nicht edilson (existiert nicht als medium)
    "pl": "pl_PL-gosia-medium",
    "nl": "nl_NL-mls-medium",
    "cs": "cs_CZ-jirka-medium",
    "el": "el_GR-rapunzelina-medium",
    "hu": "hu_HU-anna-medium",          # nicht diana (existiert nicht)
    "ro": "ro_RO-mihai-medium",
    "sv": "sv_SE-nst-medium",
    "da": "da_DK-talesyntese-medium",
    "fi": "fi_FI-harri-medium",
    "sk": "sk_SK-lili-medium",
    "bg": "bg_BG-dimitar-medium",
    "sl": "sl_SI-artur-medium",
    "et": None,  # keine Piper-Voice verfuegbar
    "lv": "lv_LV-aivars-medium",        # hat phoneme-luecken, aber lauffaehig
    "lt": None,  # keine Piper-Voice
    "hr": None,  # keine Piper-Voice
    "ga": None,  # keine Piper-Voice
    "mt": None,  # keine Piper-Voice
    "no": "no_NO-talesyntese-medium",
}


def _download_piper_voice(voice_id: str) -> tuple[Path, Path] | None:
    """Laedt eine Piper-Voice (onnx + json) von HuggingFace.

    voice_id Format: 'de_DE-thorsten-medium' -> Pfad: de/de_DE/thorsten/medium/
    """
    parts = voice_id.split("-")
    if len(parts) != 3:
        return None
    lang_region, name, quality = parts
    base = (f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang_region.split('_')[0]}/{lang_region}/{name}/{quality}/"
            f"{voice_id}.onnx")
    cache_dir = ROOT / "piper_cache"
    cache_dir.mkdir(exist_ok=True)
    onnx_path = cache_dir / f"{voice_id}.onnx"
    json_path = cache_dir / f"{voice_id}.onnx.json"
    if onnx_path.exists() and json_path.exists():
        return onnx_path, json_path
    import httpx
    with httpx.Client(follow_redirects=True, timeout=120) as c:
        for url, dest in [(base, onnx_path),
                          (base + ".json", json_path)]:
            if dest.exists():
                continue
            try:
                r = c.get(url)
                if r.status_code != 200:
                    print(f"    download failed: {url[-80:]} -> {r.status_code}", flush=True)
                    return None
                dest.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"    download err {url[-80:]}: {e}", flush=True)
                return None
    return onnx_path, json_path


def gen_piper(samples: list, errors: list) -> None:
    """Piper: ein Modell pro Sprache, schnell, CPU-only (Piper kein CUDA noetig)."""
    print("\n=== PIPER ===", flush=True)
    try:
        from piper import PiperVoice  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"  PIPER SKIP: {e}", flush=True)
        errors.append(f"piper: import {e}")
        return

    texts_data = json.loads(TEXTS.read_text(encoding="utf-8"))
    for lang, voice_id in PIPER_VOICES.items():
        if voice_id is None:
            continue
        result = _download_piper_voice(voice_id)
        if result is None:
            errors.append(f"piper/{lang}: voice download failed")
            continue
        onnx_path, json_path = result
        try:
            voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
        except Exception as e:  # noqa: BLE001
            errors.append(f"piper/{lang}/{voice_id}: load {e}")
            continue

        for variant in VARIANTS:
            text = texts_data["texts"][variant].get(lang)
            if not text:
                continue
            out_dir = AUDIO / "piper" / voice_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{lang}_{variant}.wav"
            # Wenn Sample bereits existiert, nur Metadaten ins Array aufnehmen
            if out_path.exists() and out_path.stat().st_size > 1024:
                samples.append({
                    "id": f"piper-{voice_id}-{lang}-{variant}",
                    "provider": "piper",
                    "provider_display": FRAMEWORK_DISPLAY["piper"],
                    "voice_id": voice_id,
                    "voice_name": voice_id,
                    "language": lang,
                    "language_name": next((l["name"] for l in texts_data["languages"] if l["code"] == lang), lang),
                    "variant": variant,
                    "gender": "unknown",
                    "model_type": "piper-vits",
                    "model_size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 1),
                    "audio_path": str(out_path.relative_to(OUT.parent)),
                    "provider_url": f"https://huggingface.co/rhasspy/piper-voices/tree/main/{voice_id.split('-')[0].split('_')[0]}/{voice_id.split('-')[0]}/{voice_id.split('-')[1]}/{voice_id.split('-')[2]}",
                    "license": "mit",
                    "generation_time_ms": None,
                    "sample_rate": 22050,
                })
                continue
            try:
                t0 = time.monotonic()
                audio_iter = voice.synthesize(text)
                chunks = []
                for chunk in audio_iter:
                    samples_arr = None
                    for attr in ("audio_float_array", "samples", "audio"):
                        if hasattr(chunk, attr):
                            samples_arr = np.array(getattr(chunk, attr), dtype=np.float32)
                            break
                    if samples_arr is not None and len(samples_arr) > 0:
                        chunks.append(samples_arr)
                if not chunks:
                    raise RuntimeError("kein Audio-Output")
                samples_np = np.concatenate(chunks)
                sr = getattr(chunk, "sample_rate", 22050) if chunks else 22050
                sf.write(str(out_path), samples_np, sr, format="WAV")
                ms = int((time.monotonic() - t0) * 1000)
                samples.append({
                    "id": f"piper-{voice_id}-{lang}-{variant}",
                    "provider": "piper",
                    "provider_display": FRAMEWORK_DISPLAY["piper"],
                    "voice_id": voice_id,
                    "voice_name": voice_id,
                    "language": lang,
                    "language_name": next((l["name"] for l in texts_data["languages"] if l["code"] == lang), lang),
                    "variant": variant,
                    "gender": "unknown",
                    "model_type": "piper-vits",
                    "model_size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 1),
                    "audio_path": str(out_path.relative_to(OUT.parent)),
                    "provider_url": f"https://huggingface.co/rhasspy/piper-voices/tree/main/{voice_id.split('-')[0].split('_')[0]}/{voice_id.split('-')[0]}/{voice_id.split('-')[1]}/{voice_id.split('-')[2]}",
                    "license": "mit",
                    "generation_time_ms": ms,
                    "sample_rate": sr,
                })
                print(f"  piper {lang}/{variant}: OK ({ms}ms)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(f"piper/{voice_id}/{lang}_{variant}: {e}")
                print(f"  piper {lang}/{variant}: FAIL {e}", flush=True)
        del voice


def gen_xtts(samples: list, errors: list) -> None:
    """Coqui XTTS-v2: multi-language. Laedt English-Reference + generiert 25 Sprachen."""
    print("\n=== XTTS-v2 ===", flush=True)
    try:
        from TTS.api import TTS as CoquiTTS  # type: ignore
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"  XTTS SKIP: {e}", flush=True)
        errors.append(f"xtts: import {e}")
        return

    texts_data = json.loads(TEXTS.read_text(encoding="utf-8"))
    try:
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    except Exception as e:  # noqa: BLE001
        print(f"  XTTS load failed: {e}", flush=True)
        errors.append(f"xtts: load {e}")
        return

    # Reference-Audio: irgendein kurzes English-WAV (Piper-Output)
    ref_wav = None
    for cand in (ROOT / "piper_cache").glob("en_US-lessac-medium*"):
        pass
    # Nutze erst-beste WAV im out-Ordner als Speaker-Reference
    for cand in (AUDIO / "piper").rglob("en_v1_simple.wav"):
        ref_wav = str(cand)
        break
    if not ref_wav:
        errors.append("xtts: keine Reference-Audio verfuegbar")
        return

    XTTS_LANG_MAP = {
        "de": "de", "en": "en", "fr": "fr", "es": "es", "it": "it",
        "pt": "pt", "pl": "pl", "nl": "nl", "cs": "cs", "el": "el",
        "hu": "hu", "ro": "ro", "sv": "sv", "da": "da", "fi": "fi",
        "sk": "sk", "bg": "bg", "hr": "hr", "sl": "sl", "et": "et",
        "lv": "lv", "lt": "lt", "no": "tr",  # no nicht direkt, fallback tr
        # ga, mt: nicht unterstuetzt
    }
    for lang, xtts_lang in XTTS_LANG_MAP.items():
        for variant in VARIANTS:
            text = texts_data["texts"][variant].get(lang)
            if not text:
                continue
            out_dir = AUDIO / "xtts" / "xtts_v2"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{lang}_{variant}.wav"
            if out_path.exists() and out_path.stat().st_size > 1024:
                continue
            try:
                t0 = time.monotonic()
                wav = model.tts(text=text, language=xtts_lang, speaker_wav=ref_wav)
                sf.write(str(out_path), np.array(wav, dtype=np.float32), 24000, format="WAV")
                ms = int((time.monotonic() - t0) * 1000)
                samples.append({
                    "id": f"xtts-xtts_v2-{lang}-{variant}",
                    "provider": "xtts",
                    "provider_display": FRAMEWORK_DISPLAY["xtts"],
                    "voice_id": "xtts_v2",
                    "voice_name": "XTTS-v2 (multilingual)",
                    "language": lang,
                    "language_name": next((l["name"] for l in texts_data["languages"] if l["code"] == lang), lang),
                    "variant": variant,
                    "gender": "unknown",
                    "model_type": "xtts-v2",
                    "model_size_mb": 1800,  # ca. 1.8 GB
                    "audio_path": str(out_path.relative_to(OUT.parent)),
                    "provider_url": "https://huggingface.co/coqui/XTTS-v2",
                    "license": "coqui-public-model-license",
                    "generation_time_ms": ms,
                    "sample_rate": 24000,
                })
                print(f"  xtts {lang}/{variant}: OK ({ms}ms)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(f"xtts/{lang}_{variant}: {e}")
                print(f"  xtts {lang}/{variant}: FAIL {e}", flush=True)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def gen_kokoro(samples: list, errors: list) -> None:
    """Kokoro-82M: simpel via Kokoro-Onnx oder direct via misraszymon/kokoro-onnx."""
    print("\n=== KOKORO-82M ===", flush=True)
    try:
        import soundfile as sf  # type: ignore
        import numpy as np  # type: ignore
        import httpx
    except Exception as e:  # noqa: BLE001
        print(f"  KOKORO SKIP: {e}", flush=True)
        errors.append(f"kokoro: import {e}")
        return

    # Nutze Kokoro via kokoro-onnx Python-Package
    try:
        from kokoro_onnx import Kokoro  # type: ignore
    except Exception:
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages",
                            "--quiet", "kokoro-onnx"], check=True, timeout=180)
            from kokoro_onnx import Kokoro  # type: ignore
        except Exception as e:  # noqa: BLE001
            print(f"  KOKORO install/load failed: {e}", flush=True)
            errors.append(f"kokoro: install {e}")
            return

    try:
        model = Kokoro.from_pretrained(speed=1.0)
    except Exception as e:  # noqa: BLE001
        print(f"  KOKORO load failed: {e}", flush=True)
        errors.append(f"kokoro: load {e}")
        return

    # Kokoro spricht: en, de, fr, es, it, pt, ko, ja, zh, hi
    KOKORO_LANGS = ["en", "de", "fr", "es", "it", "pt"]
    texts_data = json.loads(TEXTS.read_text(encoding="utf-8"))
    for lang in KOKORO_LANGS:
        for variant in VARIANTS:
            text = texts_data["texts"][variant].get(lang)
            if not text:
                continue
            out_dir = AUDIO / "kokoro" / "kokoro-82m"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{lang}_{variant}.wav"
            if out_path.exists() and out_path.stat().st_size > 1024:
                continue
            try:
                t0 = time.monotonic()
                samples_np = model.create(text, voice=next(iter(KOKORO_VOICES.get(lang, ["af_heart"])), "af_heart"), lang=lang)
                sf.write(str(out_path), samples_np, 24000, format="WAV")
                ms = int((time.monotonic() - t0) * 1000)
                samples.append({
                    "id": f"kokoro-kokoro-82m-{lang}-{variant}",
                    "provider": "kokoro",
                    "provider_display": FRAMEWORK_DISPLAY["kokoro"],
                    "voice_id": "kokoro-82m",
                    "voice_name": "Kokoro-82M",
                    "language": lang,
                    "language_name": next((l["name"] for l in texts_data["languages"] if l["code"] == lang), lang),
                    "variant": variant,
                    "gender": "female",
                    "model_type": "kokoro-styletts2",
                    "model_size_mb": 327,
                    "audio_path": str(out_path.relative_to(OUT.parent)),
                    "provider_url": "https://huggingface.co/hexgrad/Kokoro-82M",
                    "license": "apache-2.0",
                    "generation_time_ms": ms,
                    "sample_rate": 24000,
                })
                print(f"  kokoro {lang}/{variant}: OK ({ms}ms)", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(f"kokoro/{lang}_{variant}: {e}")
                print(f"  kokoro {lang}/{variant}: FAIL {e}", flush=True)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# Kokoro Voice-Namen pro Sprache
KOKORO_VOICES = {
    "en": ["af_heart", "af_bella", "am_adam"],
    "de": ["bf_emma", "bm_george"],
    "fr": ["bf_alice", "bm_lewis"],
    "es": ["ef_dora", "em_alex"],
    "it": ["if_isabella", "im_nicola"],
    "pt": ["pf_delfina", "pm_diogo"],
    "ja": ["jf_alpha"],
    "zh": ["zf_xiaobei"],
    "ko": ["kf_alpha"],
    "hi": ["hf_alpha", "hm_omega"],
}


def main() -> int:
    texts_data = json.loads(TEXTS.read_text(encoding="utf-8"))
    print(f"S3b: GPU-VM Generierung. Device={DEVICE}", flush=True)
    print(f"  Texte: {len(texts_data['languages'])} Sprachen × {len(VARIANTS)} Varianten", flush=True)

    samples: list[dict] = []
    errors: list[str] = []

    # 1. PIPER (schnellste, leichteste -> erste Referenzen fuer XTTS)
    gen_piper(samples, errors)

    # 2. XTTS-v2 (multi-lang, braucht Reference-Audio)
    gen_xtts(samples, errors)

    # 3. KOKORO (optional, falls Zeit bleibt)
    gen_kokoro(samples, errors)

    print(f"\n=== ZUSAMMENFASSUNG ===", flush=True)
    print(f"  Samples: {len(samples)}", flush=True)
    print(f"  Fehler:  {len(errors)}", flush=True)
    for e in errors[:20]:
        print(f"    - {e}")

    META.write_text(json.dumps({
        "_meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": DEVICE,
            "frameworks": ["piper", "xtts", "kokoro"],
        },
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {META}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
