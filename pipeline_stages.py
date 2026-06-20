"""
pipeline_stages.py
==================

The four pipeline stages, ported faithfully from the original R scripts and the
Jupyter notebook. Prompts, JSON schemas, parsers, temperatures, token budgets
and the label-order shuffle are reproduced as-is. Only the *runtime* (Python
instead of R/notebook) and the calling interface have changed.

Each stage is a plain function: text in, text/structured-out. No file I/O and
no orchestration live here — that is run_pipeline.py's job.

A `--mock` path (cfg.mock=True) stubs every network call with deterministic
output so the DAG, caching and output schema can be validated offline without
API keys or spend.
"""

from __future__ import annotations

import json
import re
import threading
import time
from hashlib import sha256
from random import Random

import pipeline_config as C

# ---------------------------------------------------------------------------
# Lazy, cached SDK / tokenizer handles
# ---------------------------------------------------------------------------
_openai_clients: dict[str, object] = {}
_encodings: dict[str, object] = {}
_throttle_lock = threading.Lock()
_last_call_at = [0.0]


def _throttle(rps: float) -> None:
    """Global per-process rate gate (mirrors httr2 req_throttle)."""
    if rps <= 0:
        return
    min_interval = 1.0 / rps
    with _throttle_lock:
        wait = min_interval - (time.monotonic() - _last_call_at[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_at[0] = time.monotonic()


def _get_encoding(name: str):
    if name not in _encodings:
        import tiktoken
        _encodings[name] = tiktoken.get_encoding(name)
    return _encodings[name]


def _openai_client(api_key: str):
    if api_key not in _openai_clients:
        from openai import OpenAI
        _openai_clients[api_key] = OpenAI(api_key=api_key)
    return _openai_clients[api_key]


# ---------------------------------------------------------------------------
# Low-level chat call with retry + throttle, dispatched by provider.
# Returns the raw text content of the model's reply.
# ---------------------------------------------------------------------------
def _chat(cfg, model, *, system=None, user, max_tokens=None, json_schema=None,
          want_json=False):
    provider = C.provider_for_model(model)
    _throttle(cfg.throttle_rps)

    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            if provider == "openai":
                return _openai_call(cfg, model, system, user, max_tokens, json_schema)
            if provider == "anthropic":
                return _anthropic_call(cfg, model, user, max_tokens)
            if provider == "google":
                return _gemini_call(cfg, model, user, want_json)
            if provider == "dashscope":
                return _dashscope_call(cfg, model, system, user, json_schema)
            raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:  # transient -> backoff, mirrors req_retry(max_tries=5)
            last_err = e
            if attempt == cfg.max_retries or not _is_transient(e):
                raise
            time.sleep(2 ** attempt)  # 2,4,8,16,32s
    raise last_err  # pragma: no cover


def _is_transient(e) -> bool:
    msg = str(e).lower()
    for code in ("429", "500", "503", "529", "timeout", "timed out",
                 "rate limit", "overloaded", "connection"):
        if code in msg:
            return True
    return False


def _temp_kwargs(cfg) -> dict:
    return {} if cfg.drop_temperature else {"temperature": cfg.temperature}


def _openai_call(cfg, model, system, user, max_tokens, json_schema):
    client = _openai_client(C.get_api_key("openai"))
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs = dict(model=model, messages=messages, timeout=cfg.request_timeout,
                  **_temp_kwargs(cfg))
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    if json_schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _anthropic_call(cfg, model, user, max_tokens):
    # Mirrors call_claude_api / build_req_claude in the R scripts.
    import requests
    body = {"model": model, "max_tokens": max_tokens or 1024,
            "messages": [{"role": "user", "content": user}]}
    if not cfg.drop_temperature:
        body["temperature"] = cfg.temperature
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": C.get_api_key("anthropic"),
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=body, timeout=cfg.request_timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:200]}")
    text = r.json()["content"][0]["text"]
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text)


def _gemini_call(cfg, model, user, want_json=False):
    # Mirrors call_gemini_api / build_req_gemini in the R scripts.
    # call_gemini_api (moral gen) set responseMimeType="application/json";
    # build_req_gemini (labeling) did not -> want_json toggles it.
    import requests
    gen = {} if cfg.drop_temperature else {"temperature": cfg.temperature}
    if want_json:
        gen["responseMimeType"] = "application/json"
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": C.get_api_key("google"),
                 "content-type": "application/json"},
        json={"contents": [{"parts": [{"text": user}]}], "generationConfig": gen},
        timeout=cfg.request_timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _dashscope_call(cfg, model, system, user, json_schema):
    # Mirrors call_qwen_api in storyMorals_API.R (OpenAI-compatible endpoint).
    import requests
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body = {"model": model, "messages": messages}
    if not cfg.drop_temperature:
        body["temperature"] = cfg.temperature
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    r = requests.post(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {C.get_api_key('dashscope')}",
                 "Content-Type": "application/json"},
        json=body, timeout=cfg.request_timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Qwen {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def read_text_file(path) -> str:
    """UTF-8 read with replacement fallback (notebook read_text_file)."""
    from pathlib import Path
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")


def load_label_list(cfg) -> list[str]:
    """Labels = column `TAXONOMY_LABEL_COLUMN` of the taxonomy CSV, verbatim."""
    import csv
    with open(cfg.taxonomy_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    col = C.TAXONOMY_LABEL_COLUMN
    return [r[col] for r in body if r and r[col].strip()]


def derive_seed(cfg, book_id, condition, moral_model, moral_index, run) -> int:
    """Deterministic per-row label-shuffle seed derived from the master seed.

    The R script drew a random sample.int seed and *saved it* so resume reused
    the identical shuffle. We get the same reproducibility deterministically.
    The seed depends on the moral's identity (book, condition, moral_model,
    moral_index) and the run — but NOT on the labeling model, so all label
    models share one shuffle per (moral, run), matching the R pipeline.
    """
    key = f"{cfg.run_seed}|{book_id}|{condition}|{moral_model}|{moral_index}|{run}"
    h = sha256(key.encode())
    return int.from_bytes(h.digest()[:4], "big")  # 32-bit, like .Machine$integer.max


def shuffled_labels(label_list, order_seed) -> list[str]:
    """One label permutation for a given seed (R: withr::with_seed(seed, sample))."""
    labels = list(label_list)
    Random(order_seed).shuffle(labels)
    return labels


# ---------------------------------------------------------------------------
# Stage A — Chunk summary  (chunkSummary_API_GPT.ipynb)
# ---------------------------------------------------------------------------
def chunk_text_by_tokens(text, cfg) -> list[str]:
    enc = _get_encoding(cfg.token_encoding)
    ids = enc.encode(text)
    return [enc.decode(ids[s:s + cfg.chunk_size_tokens])
            for s in range(0, len(ids), cfg.chunk_size_tokens)]


def stage_chunk_summary(full_text, cfg) -> str:
    """Token-chunk the text, summarize each chunk, join with CHUNK_JOINER."""
    chunks = chunk_text_by_tokens(full_text, cfg)
    if cfg.mock:
        return C.CHUNK_JOINER.join(
            f"[MOCK] Ten key events of chunk {i + 1}/{len(chunks)}."
            for i in range(len(chunks))
        )
    summaries = []
    for chunk in chunks:
        user = C.CHUNK_USER_PROMPT_TEMPLATE.format(chunk_text=chunk)
        summaries.append(_chat(cfg, cfg.chunk_model, system=C.CHUNK_SYSTEM_PROMPT,
                               user=user, max_tokens=cfg.chunk_max_tokens))
    return C.CHUNK_JOINER.join(summaries)


# ---------------------------------------------------------------------------
# Stage B — Short summary / condenser  (ChunkSummaryCondenser_API.R)
# ---------------------------------------------------------------------------
def stage_short_summary(chunk_summary, cfg) -> str:
    """Condense the chunk summary into one paragraph. System=prompt, user=text."""
    if cfg.mock:
        return "[MOCK] One-paragraph condensed summary of the plot events."
    return _chat(cfg, cfg.short_model, system=C.SHORT_SUMMARY_PROMPT,
                 user=chunk_summary)


# ---------------------------------------------------------------------------
# Stage C — Moral generation  (storyMorals_API.R)
# ---------------------------------------------------------------------------
def build_moral_prompt(input_text, language, input_type) -> str:
    """Verbatim port of storyMorals_API.R::build_prompt (fulltext/summary)."""
    opening = ("Given the following novel, " if input_type == "fulltext"
               else "Given the following plot summary, ")
    body_label = "Novel:\n" if input_type == "fulltext" else "Plot Summary:\n"
    return (
        opening
        + "generate three story morals as pithy, memorable statements that are "
        + "a single phrase or sentence IN " + language + ". "
        + "The three morals must be meaningfully different (not paraphrases). "
        + body_label
        + input_text + "\n\n"
        + "Return your response as a JSON object with this exact structure:\n"
        + '{\n'
        + '  "pithy_morals": [\n'
        + '    "first pithy moral",\n'
        + '    "second pithy moral",\n'
        + '    "third pithy moral"\n'
        + '  ]\n'
        + '}'
    )


def _parse_morals(raw) -> list[str]:
    """Parse {"pithy_morals":[...]} -> list (R: fromJSON(x)$pithy_morals)."""
    if not raw:
        return []
    txt = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        obj = json.loads(txt)
        return [str(m) for m in obj.get("pithy_morals", [])]
    except Exception:
        return []


def stage_generate_morals(input_text, input_type, language, model, cfg) -> list[str]:
    """Return the list of pithy morals (length n_morals when well-formed)."""
    if cfg.mock:
        return [f"[MOCK] Moral {i + 1} for a {input_type} input ({language}) "
                f"by {model}." for i in range(cfg.n_morals)]
    prompt = build_moral_prompt(input_text, language, input_type)
    provider = C.provider_for_model(model)
    # GPT/Qwen enforce the schema; Gemini uses responseMimeType=json (want_json).
    schema = C.MORAL_JSON_SCHEMA if provider in ("openai", "dashscope") else None
    raw = _chat(cfg, model, system=C.MORAL_SYSTEM_PROMPT, user=prompt,
                json_schema=schema, want_json=True)
    morals = _parse_morals(raw)
    time.sleep(cfg.moral_delay_seconds)  # storyMorals_API.R delay_seconds
    return morals


# ---------------------------------------------------------------------------
# Stage D — Value labeling  (storyMorals_ValueExtraction_API.R)
# ---------------------------------------------------------------------------
def build_label_prompt(moral_text, labels) -> str:
    """Verbatim port of storyMorals_ValueExtraction_API.R::build_prompt."""
    label_block = "\n- ".join(labels)
    return (
        "You are annotating the moral of a story using a fixed taxonomy of values.\n\n"
        "Select ALL labels from the taxonomy that apply. Multiple labels are allowed. "
        "Use ONLY labels from this list (verbatim):\n\n"
        f"- {label_block}\n\n"
        f'Moral to annotate:\n"""\n{moral_text}\n"""\n\n'
        'Return ONLY a JSON array of the chosen label strings, e.g. ["Label A", "Label B"]. '
        "No prose, no explanation, no markdown."
    )


def extract_json_array(txt):
    """Port of extract_json_array: strip fences, keep LAST [...] block, parse."""
    if not txt or not isinstance(txt, str) or not txt.strip():
        return None
    txt = re.sub(r"```(json)?", "", txt)
    arrays = re.findall(r"\[.*?\]", txt, flags=re.S)  # non-greedy, all arrays
    if not arrays:
        return None
    try:
        parsed = json.loads(arrays[-1])               # keep the last one
    except Exception:
        return None
    return [str(x) for x in parsed]


def stage_label_moral(moral_text, label_list, order_seed, model, cfg):
    """Assign taxonomy labels to one moral under one label-order shuffle.

    Returns a list of labels (possibly empty), or None on hard failure.
    """
    labels = shuffled_labels(label_list, order_seed)
    if cfg.mock:
        # Deterministic 2-label mock keyed by the shuffle seed.
        rng = Random(order_seed)
        return sorted(rng.sample(label_list, 2))
    raw = _chat(cfg, model, user=build_label_prompt(moral_text, labels),
                max_tokens=1024)
    return extract_json_array(raw)
