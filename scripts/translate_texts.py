#!/usr/bin/env python3
"""M51 S1: Uebersetzt die 3 Ausgangstexte (deutsch) in 24 weitere Sprachen.

Pro Sprache EINE eigene LLM-Anfrage, die alle 3 Varianten in einem Call
uebersetzt (statt 75 Einzel-Calls). Spart Quota und erhoeht Konsistenz, weil
die Sprachregister innerhalb einer Sprache einheitlich bleiben.

Provider-Weiche (Pflicht nach Bauplan):
  1. Versuch: Gemini 3.1 Pro Preview (Vault-Slot 'gemini').
  2. Fallback: OpenAI gpt-4o-mini (Vault-Slot 'openai') -- wenn Gemini-Quota
     erschöpft ist (HTTP 429). Dev-Brain nutzt OpenAI ohnehin als
     Default-LLM-Provider, also liegt der Key verlaesslich im Vault.

Beide Slots kommen aus dem Dev-Brain Vault, darum muss dieses Script mit
Zugriff auf /root/dev-brain/backend laufen.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

DEVBRAIN_BACKEND = Path("/root/dev-brain/backend")
sys.path.insert(0, str(DEVBRAIN_BACKEND))

import secrets_vault  # noqa: E402  -- Pfad oben gesetzt

ROOT = Path(__file__).resolve().parent.parent
TEXTS_JSON = ROOT / "data" / "texts.json"

GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

CONCURRENCY = 3
MAX_RETRIES = 3

VARIANTS = ["v1_simple", "v2_medium", "v3_emotional"]


def _prompt(lang_name: str, lang_code: str, texts_de: dict[str, str]) -> str:
    return (
        f"Uebersetze die folgenden drei Texte in {lang_name} "
        f"(Sprachcode: {lang_code}).\n"
        f"Die Texte werden fuer ein Text-to-Speech-Voice-Modell vorgelesen.\n\n"
        f"Anforderungen:\n"
        f"- Natuerliche, fluessige Sprache -- optimiert fuer VORLESEN, nicht Lesen.\n"
        f"- Keine komplizierten Schachtelsaetze.\n"
        f"- Zahlen und Daten in der lokalen Konvention.\n"
        f"- Eigennamen bleiben in der lokalen Form.\n"
        f"- Bewahre den emotionalen Ton (v3 muss begeistert klingen).\n\n"
        f"Antworte AUSSCHLIESSLICH als JSON-Objekt mit genau drei Schluesseln "
        f"(v1_simple, v2_medium, v3_emotional) und den uebersetzten Strings als "
        f"Werte. Kein Markdown, keine Erklaerung.\n\n"
        f"Texte:\n"
        f"v1_simple: {texts_de['v1_simple']}\n"
        f"v2_medium: {texts_de['v2_medium']}\n"
        f"v3_emotional: {texts_de['v3_emotional']}\n"
    )


async def _gemini_one(
    client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
) -> dict[str, str]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    r = await client.post(
        GEMINI_URL, params={"key": api_key}, json=payload, timeout=60.0,
    )
    if r.status_code != 200:
        raise _QuotaError(r.status_code, r.text[:200])
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


async def _openai_one(
    client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
) -> dict[str, str]:
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "Du bist ein professioneller Uebersetzer. Antworte NUR mit JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    r = await client.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60.0,
    )
    if r.status_code != 200:
        raise _QuotaError(r.status_code, r.text[:200])
    data = r.json()
    return json.loads(data["choices"][0]["message"]["content"])


class _QuotaError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


async def _translate_one(
    client: httpx.AsyncClient,
    providers: list[tuple[str, str]],
    prompt: str,
) -> dict[str, str]:
    """Versucht alle Provider der Reihe nach. Wirft, wenn alle versagen."""
    last_err: Exception | None = None
    for name, key in providers:
        fn = _gemini_one if name == "gemini" else _openai_one
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                parsed = await fn(client, key, prompt)
                if all(v in parsed and isinstance(parsed[v], str) and parsed[v].strip()
                       for v in VARIANTS):
                    return {v: parsed[v].strip() for v in VARIANTS}
                last_err = RuntimeError(f"{name}: unvollstaendig {list(parsed.keys())}")
            except _QuotaError as e:
                last_err = e
                if e.status in (429, 401, 403):
                    break  # Provider-Wechsel, kein Retry
            except Exception as e:  # noqa: BLE001
                last_err = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
    raise RuntimeError(f"Alle Provider versagt. Letzter Fehler: {last_err}")


async def _producer(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    providers: list[tuple[str, str]],
    lang: dict,
    texts_de: dict[str, str],
    out: dict,
    errors: list,
    progress: list,
):
    async with sem:
        try:
            prompt = _prompt(lang["name"], lang["code"], texts_de)
            translations = await _translate_one(client, providers, prompt)
            for variant, text in translations.items():
                out[variant][code := lang["code"]] = text
            progress.append(1)
            done = len(progress)
            print(f"  [{done:>2}/24] {code:>2} -- {lang['name']}", flush=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{lang['code']}: {e}")
            print(f"  [FEHLER] {lang['code']}: {e}", flush=True)


async def main() -> int:
    data = json.loads(TEXTS_JSON.read_text(encoding="utf-8"))
    texts_de = {v: data["texts"][v]["de"] for v in VARIANTS}
    out = {v: {"de": texts_de[v]} for v in VARIANTS}

    targets = [l for l in data["languages"] if l["code"] != "de"]
    print(f"S1: Uebersetze 3 Varianten × {len(targets)} Sprachen "
          f"(Concurrency={CONCURRENCY})", flush=True)

    providers: list[tuple[str, str]] = []
    for slot in ("gemini", "openai"):
        try:
            key = await secrets_vault.get_secret(slot, "api_key")
            if key:
                providers.append((slot, key))
                print(f"  Provider: {slot} (key len={len(key)})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  Provider {slot} nicht verfuegbar: {e}", flush=True)
    if not providers:
        print("FEHLER: Weder Gemini- noch OpenAI-Key im Vault.", flush=True)
        return 2

    sem = asyncio.Semaphore(CONCURRENCY)
    errors: list[str] = []
    progress: list[int] = []
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[
            _producer(sem, client, providers, lang, texts_de, out, errors, progress)
            for lang in targets
        ])

    duration = time.monotonic() - t0
    print(f"\nFertig in {duration:.1f}s. Erfolge={len(progress)}/{len(targets)}, "
          f"Fehler={len(errors)}", flush=True)

    if errors:
        print("\nFehler-Detail:")
        for e in errors:
            print(f"  - {e}")

    if len(progress) < int(len(targets) * 0.8):
        print("Zuviele Fehler -- texts.json NICHT geschrieben.", flush=True)
        return 1

    data["texts"] = out
    data["_meta"]["translated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    data["_meta"]["translation_provider"] = providers[0][0]
    TEXTS_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"texts.json aktualisiert: {TEXTS_JSON}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
