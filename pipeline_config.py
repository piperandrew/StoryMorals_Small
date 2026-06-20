"""
pipeline_config.py
==================

Central configuration for the Book-to-Moral cultural-comparison pipeline.

Every parameter that the original R scripts / Jupyter notebook hard-coded is
surfaced here as a *named* config field (no silent inline defaults). The prompt
strings, JSON schemas, temperatures, token budgets, throttle/concurrency
constants and label-shuffle seeding are reproduced VERBATIM from the source
stage scripts so the ported Python behaves identically:

  - Stage A (chunk summary)  <- chunkSummary_API_GPT.ipynb
  - Stage B (short summary)  <- storyMorals_ChunkSummaryCondenser_API.R
  - Stage C (moral gen)      <- storyMorals_API.R
  - Stage D (value labeling) <- storyMorals_ValueExtraction_API.R

To run with more K (label-order randomizations) or more models, change
`Config.k` and `Config.labeling_models` — nothing else needs to move.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "CulturalComparisonSample"   # flat dir of .txt files
WORK_DIR = PROJECT_ROOT / "work"                         # cached intermediates
OUTPUT_DIR = PROJECT_ROOT / "output"                     # final deliverables
TAXONOMY_CSV = PROJECT_ROOT / "Values_Taxonomy.csv"

# The three input conditions produced per book. NOTE: the "ensemble"
# (chunk ∪ short) condition is intentionally NOT generated here — it is a
# downstream R row-union of core labels.
CONDITIONS = ("full_text", "chunk_summary", "short_summary")


# ---------------------------------------------------------------------------
# Stage A — Chunk summary  (verbatim from chunkSummary_API_GPT.ipynb)
# ---------------------------------------------------------------------------
CHUNK_SYSTEM_PROMPT = "You are a helpful assistant that summarizes literature."
CHUNK_USER_PROMPT_TEMPLATE = (
    "Here is a portion of a novel. Please summarize it IN ENGLISH by listing "
    "the TEN most significant events and plot developments:\n\n{chunk_text}"
)
TOKEN_ENCODING = "o200k_base"     # tiktoken encoding used in the notebook
CHUNK_SIZE_TOKENS = 14000         # fixed-size token chunks
CHUNK_MAX_TOKENS = 1024           # max_completion_tokens per chunk summary
CHUNK_JOINER = "\n\n"             # how per-chunk summaries are concatenated


# ---------------------------------------------------------------------------
# Stage B — Short summary / condenser  (from ChunkSummaryCondenser_API.R)
# ---------------------------------------------------------------------------
# The R call-site (not the function default) is the source of truth: it used
# model="gpt-5.4", temperature=0, and the one-paragraph prompt below.
SHORT_SUMMARY_PROMPT = (
    "Please provide a 1 paragraph summary of the following plot events of a novel:"
)


# ---------------------------------------------------------------------------
# Stage C — Moral generation  (from storyMorals_API.R: build_prompt)
# ---------------------------------------------------------------------------
MORAL_SYSTEM_PROMPT = "You are a helpful assistant. Always respond in JSON format."

# JSON schema enforced on the GPT response (storyMorals_API.R: call_gpt_api).
MORAL_JSON_SCHEMA = {
    "name": "pithy_morals",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "pithy_morals": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["pithy_morals"],
        "additionalProperties": False,
    },
}
N_MORALS = 3  # moral_index 1..3


# ---------------------------------------------------------------------------
# Stage D — Value labeling  (from storyMorals_ValueExtraction_API.R)
# ---------------------------------------------------------------------------
# Labels come from column 1 (rokeach_value) of the taxonomy, verbatim.
TAXONOMY_LABEL_COLUMN = 0  # 0-based: first column


# ---------------------------------------------------------------------------
# Default models per stage
# ---------------------------------------------------------------------------
# Stages A/B (summaries) use the cheaper mini; stages C/D (morals + labels) run
# the full cross of these two models. A global `--model X` overrides all of them.
DEFAULT_CHUNK_MODEL = "gpt-5.4-mini"
DEFAULT_SHORT_MODEL = "gpt-5.4-mini"
DEFAULT_MORAL_MODELS = ["gpt-5.4", "gemini-3.1-pro-preview"]
DEFAULT_LABELING_MODELS = ["gpt-5.4", "gemini-3.1-pro-preview"]


# ---------------------------------------------------------------------------
# Provider routing + API keys
# ---------------------------------------------------------------------------
# Model-name prefix -> provider. Used to pick the right endpoint/key when the
# labeling_models list is extended beyond gpt-5.4.
def provider_for_model(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("qwen"):
        return "dashscope"
    raise ValueError(f"Cannot infer provider for model '{model}'")


_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "dashscope": "DASHSCOPE_API_KEY",
}


def get_api_key(provider: str) -> str:
    names = _ENV_KEY[provider]
    if isinstance(names, str):
        names = (names,)
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(
        f"No API key found for provider '{provider}'. Set ${names[0]}."
    )


# ---------------------------------------------------------------------------
# Main config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- global override: if set, forces this one model on EVERY stage ---
    model: str | None = None

    # Per-stage models. Stages A/B are single-model (summaries); stages C/D run
    # the full cross of their model lists. Empty/None -> the DEFAULT_* above.
    chunk_model: str | None = None       # Stage A  -> DEFAULT_CHUNK_MODEL
    short_model: str | None = None       # Stage B  -> DEFAULT_SHORT_MODEL
    moral_models: list[str] = field(default_factory=list)     # Stage C
    labeling_models: list[str] = field(default_factory=list)  # Stage D

    # --- generation determinism ---
    temperature: float = 0.0          # all stages run temperature=0 in the scripts
    # Set drop_temperature=True for gpt-5.x "thinking" models that reject the
    # temperature field (see comment in storyMorals_ValueExtraction_API.R).
    drop_temperature: bool = False

    # --- Stage A token chunking ---
    token_encoding: str = TOKEN_ENCODING
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS
    chunk_max_tokens: int = CHUNK_MAX_TOKENS

    # --- Stage C ---
    language: str = "English"         # output language for the morals (all books)
    n_morals: int = N_MORALS
    moral_delay_seconds: float = 1.0  # pause between moral-generation calls

    # --- Stage D ---
    k: int = 1                        # label-order randomizations per moral
    run_seed: int = 20240601          # master seed; per-row seeds derived from it
    throttle_rps: float = 5.0         # THROTTLE_RPS
    max_active: int = 8               # MAX_ACTIVE concurrent in-flight requests

    # --- network robustness (mirrors req_retry/req_timeout in the scripts) ---
    max_retries: int = 5
    request_timeout: int = 60

    # --- paths ---
    input_dir: Path = INPUT_DIR
    work_dir: Path = WORK_DIR
    output_dir: Path = OUTPUT_DIR
    taxonomy_csv: Path = TAXONOMY_CSV
    conditions: tuple = CONDITIONS

    # --- run controls ---
    mock: bool = False                # stub all API calls (offline plumbing test)
    rebuild: bool = False             # ignore completed-book skip; rebuild table
                                      # from caches (use after changing K/models)

    def __post_init__(self):
        # Per-stage defaults (used when not explicitly provided).
        self.chunk_model = self.chunk_model or DEFAULT_CHUNK_MODEL
        self.short_model = self.short_model or DEFAULT_SHORT_MODEL
        if not self.moral_models:
            self.moral_models = list(DEFAULT_MORAL_MODELS)
        if not self.labeling_models:
            self.labeling_models = list(DEFAULT_LABELING_MODELS)
        # Global override: --model X collapses every stage onto one model.
        if self.model:
            self.chunk_model = self.model
            self.short_model = self.model
            self.moral_models = [self.model]
            self.labeling_models = [self.model]
        self.input_dir = Path(self.input_dir)
        self.work_dir = Path(self.work_dir)
        self.output_dir = Path(self.output_dir)
        self.taxonomy_csv = Path(self.taxonomy_csv)
        # Keep mock artifacts in separate dirs so a stubbed test run can never
        # poison the cache of a real (API-backed) run on the same books.
        if self.mock:
            self.work_dir = self.work_dir.with_name(self.work_dir.name + "_mock")
            self.output_dir = self.output_dir.with_name(self.output_dir.name + "_mock")
