# Book-to-Moral pipeline

A command-line workflow that reads the moral values out of books at scale.

**Input.** A folder of plain-text books — one `.txt` file per book. Each file's
name encodes its `book_id` (the filename) and a `culture` code (the prefix
before the first underscore, e.g. `DE_…`, `JP_…`).

**What it does.** For every book, the pipeline:

1. **Summarizes** the full text — first into a chunk-by-chunk plot summary, then
   condenses that into a one-paragraph summary.
2. **Generates morals** — three short, pithy "morals of the story" from each of
   three views of the book: its *full text*, its *chunk summary*, and its *short
   summary*.
3. **Labels values** — tags each moral with values from a fixed 62-label
   taxonomy (e.g. *Care*, *Justice*, *Loyalty*).

Multiple models can run each step, so you can compare how different models read
the same book. Every step is cached, so reruns resume where they left off.

**Output.** Tidy tables (CSV + Parquet): a long table with one row per assigned
value (book × view × moral × value), a collapsed one-row-per-moral table with
its unique values, and a per-book table of the summaries themselves.

---

## Files

| File | Role |
|------|------|
| `pipeline_config.py` | All parameters plus the prompts/JSON schemas. The single place to change models, K, language, throttle, paths. |
| `pipeline_stages.py` | The four stages (text in → text/structured out), plus the API call layer (retry + throttle) and the `--mock` stubs. |
| `run_pipeline.py` | The orchestrator: discovery, the per-book DAG, caching, checkpoint/resume, fail-soft error handling, progress logging, and the final table. CLI entry point. |
| `Values_Taxonomy.csv` | The value taxonomy. Column 1 (`rokeach_value`, 62 labels) is the label set; column 2 (`schwartz_value`) is the downstream grouping. |
| `requirements.txt` | Python dependencies. |
| `example/` | A synthetic, copyright-free book + illustrative sample output (see `example/README.md`). |

> **Inputs and outputs are not in this repo.** The book corpus is copyrighted,
> and the `work/`/`output/` artifacts are derived from it; both are gitignored.
> Supply your own `.txt` files via `--input-dir`. The `example/` folder is the
> only bundled, runnable sample.

### Stages and default models

| Stage | Does | Default model(s) |
|-------|------|-------------------|
| A. Chunk summary | token-chunks the full text, summarizes each chunk | `gpt-5.4-mini` |
| B. Short summary | condenses the chunk summary to one paragraph | `gpt-5.4-mini` |
| C. Moral generation | writes 3 morals per view | `gpt-5.4` **and** `gemini-3.1-pro-preview` |
| D. Value labeling | tags each moral with taxonomy values | `gpt-5.4` **and** `gemini-3.1-pro-preview` |

Stages C and D run a **full cross**: each moral-generation model produces its own
3 morals per condition, and **every labeling model labels every moral** (so
gpt-5.4 labels gemini's morals and vice-versa). The output distinguishes
`moral_model` (who wrote the moral) from `label_model` (who assigned the value).
A global `--model X` collapses all stages onto one model.

---

## The DAG (per book)

```
full_text ─┬───────────────────────────────► condition: full_text
           │
           └─► chunk_summary ─┬─────────────► condition: chunk_summary
                              │
                              └─► short_summary  condition: short_summary
```

`short_summary` is condensed from `chunk_summary` (NOT from the full text).
For **each** of the three conditions: generate 3 candidate morals, then label
each moral against the taxonomy (K order-randomized runs per labeling model).

> The **ensemble** (chunk ∪ short) condition is intentionally **not** produced
> here — it is computed downstream in R as a row-union of core labels.

`book_id` = the `.txt` filename stem. `culture` = the filename prefix before the
first underscore. The input directory is **flat** (no per-culture
subdirectories); culture is read from the prefix — e.g. `DE_…`, `JP_…`, `IN_…`
yield cultures `DE`, `JP`, `IN`.

---

## How to run

### 0. Install deps
```bash
pip install -r requirements.txt
```
(`openai`, `tiktoken`, `pandas`, `pyarrow`, `requests`.)

### Try the bundled example first (no corpus needed)
```bash
python3 run_pipeline.py --mock --input-dir example \
    --output-dir example/demo --work-dir example/demo_work
```
See `example/README.md`. Pre-generated illustrative output is in
`example/sample_output/`.

### 1. Set the API keys
The default config uses OpenAI **and** Google models, so set both:
```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...          # or GOOGLE_API_KEY
```
(`ANTHROPIC_API_KEY` / `DASHSCOPE_API_KEY` only needed if you add claude-/qwen-
models. A single-model run via `--model gpt-5.4` needs only `OPENAI_API_KEY`.)

### 2. Dry run with no keys / no spend — validate plumbing & schema
```bash
python3 run_pipeline.py --mock --limit 1
```
Mock stubs every API call with deterministic output and writes to
`work_mock/` + `output_mock/` so it never collides with real runs.
(Use `--books <book_stem>` to target a specific file.)

### 3. Single-book end-to-end test (real API)
```bash
python3 run_pipeline.py --limit 1            # or --books <book_stem>
```
Inspect `work/<culture>/<book_stem>/` and `output/book_to_moral_long.csv`
before scaling up.

### 4. Full corpus
```bash
python3 run_pipeline.py
```
Resumable: rerun the same command and completed work is skipped.

---

## Options (`python3 run_pipeline.py --help`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | _(unset)_ | Global override: force ONE model on **all** stages (otherwise the per-stage defaults below apply). |
| `--chunk-model` | `gpt-5.4-mini` | Stage A model. |
| `--short-model` | `gpt-5.4-mini` | Stage B model. |
| `--moral-models` | `gpt-5.4,gemini-3.1-pro-preview` | Comma-separated Stage C generators. Each adds a `moral_model`. |
| `--labeling-models` | `gpt-5.4,gemini-3.1-pro-preview` | Comma-separated Stage D labelers (e.g. add `claude-sonnet-4-6`). Each adds a `label_model`; full cross with the generators. |
| `--k` | `1` | Label-order randomizations per moral. K>1 adds rows (one `run_seed` each). |
| `--language` | `English` | Moral output language. Use `__native__` to map each culture to a language via the `native_language` dict in `pipeline_config.py` (cultures not in the map fall back to English). |
| `--run-seed` | `20240601` | Master seed. Per-row label-shuffle seeds are derived deterministically from it (reproducible across reruns). |
| `--throttle-rps` | `5` | Global request rate cap (from `THROTTLE_RPS`). |
| `--max-active` | `8` | Concurrency budget (from `MAX_ACTIVE`); see note below. |
| `--drop-temperature` | off | Omit the `temperature` field (for gpt-5.x "thinking" models that reject it). |
| `--conditions` | `full_text,chunk_summary,short_summary` | Which conditions to build. |
| `--books` | all | Comma-separated stems/filenames to run. |
| `--culture` | all | Comma-separated culture codes (the filename prefixes, e.g. `DE,JP,IN`). |
| `--limit N` | none | Process at most N books. |
| `--mock` | off | Stub all API calls; write to `*_mock` dirs. |
| `--rebuild` | off | Ignore the completed-book skip; rebuild the whole table from the `work/` cache (use after changing `--k`/`--labeling-models`). |
| `--input-dir` / `--work-dir` / `--output-dir` / `--taxonomy` | see config | Path overrides. |

Reused constants live in `pipeline_config.py`: `CHUNK_SIZE_TOKENS=14000`,
`TOKEN_ENCODING="o200k_base"`, `CHUNK_MAX_TOKENS=1024`, `temperature=0`,
`THROTTLE_RPS=5`, `MAX_ACTIVE=8`, `max_retries=5`, `request_timeout=60`,
`moral_delay_seconds=1`.

---

## Outputs

### Final deliverable — `output/book_to_moral_long.csv` (+ `.parquet`)
One row per assigned value:

| column | notes |
|--------|-------|
| `book_id` | filename stem |
| `culture` | filename prefix code (e.g. `DE`, `JP`, `IN`) |
| `input_condition` | `full_text` \| `chunk_summary` \| `short_summary` |
| `moral_index` | `1`–`3` |
| `moral_text` | the moral string |
| `value_label` | a taxonomy label, **or `NA`** when the moral got zero labels |
| `moral_model` | which model **generated** the moral |
| `label_model` | which model **assigned** the value |
| `run_seed` | the per-row label-shuffle seed (shared across label models for a given moral+run) |
| `timestamp` | when the labels were produced (UTC ISO) |

- A moral with N labels → N rows.
- A moral with **zero** labels → **one** row with `value_label = NA` (empty
  labelings are preserved, not silently dropped).
- Full cross: each `(moral_model, moral_index)` is labeled by every
  `label_model`; with `--k > 1` there is one block per `(…, run)`.

### Collapsed table — `output/book_to_moral_unique.csv` (+ `.parquet`)
One row per **moral** (`moral_text` × `moral_model`), with `value_label` = the
**distinct** value labels pooled across **all** label models for that moral,
comma-separated. Drops `label_model`, `run_seed`, `timestamp`.

| column | notes |
|--------|-------|
| `book_id`, `culture`, `input_condition`, `moral_index`, `moral_text` | as in the long table |
| `value_label` | comma-separated unique labels across label models (empty if none) |
| `moral_model` | which model generated the moral |

Derived from the long table on every run (also works on an old single-model
table, where `model` is treated as the generator).

### Summaries table — `output/book_summaries.csv`
One row per book, for auditing the inputs to the moral stage:

| column | notes |
|--------|-------|
| `book_id` | filename stem |
| `culture` | filename prefix code (e.g. `DE`, `JP`, `IN`) |
| `chunk_summary` | the aggregated Stage A chunk summary |
| `short_summary` | the Stage B one-paragraph condensation |

### Cached intermediates (auditable, resumable) — under `work/{culture}/{book_id}/`
```
chunk_summary.txt          # Stage A output
short_summary.txt          # Stage B output
stage_models.json          # which model produced each summary (+ input hash)
morals_{condition}.json    # Stage C: {input_hash, language, morals:{<moral_model>:{pithy_morals:[...], timestamp}}}
values_{condition}.json    # Stage D: {k, moral_models, label_models, morals_hash,
                           #           records:[{moral_model, moral_index, label_model, run, run_seed, labels, timestamp}]}
```

### `output/errors.log`
Tab-separated `timestamp · book_id · stage · message` for any book that errored.
One bad book never aborts the corpus.

---

## Robustness

- **Checkpoint/resume (two layers).**
  1. *Output checkpoint:* the final tables (`book_to_moral_long.csv`,
     `book_summaries.csv`, `.parquet`) are rewritten **after every book**, via a
     `.tmp`-then-rename so a kill mid-write never corrupts them.
  2. *Stage cache:* every intermediate is cached under `work/` keyed by book and
     stage; any stage whose artifact exists and is non-empty is skipped. The
     cache is **model- and content-aware**: a stage regenerates if its model
     changed (recorded in `work/.../stage_models.json` / the artifact) **or** if
     its upstream input text changed (content hash) — so a new chunk model
     cascades correctly through summaries → morals → labels.
  3. *Config guard:* a `run_config.json` is written to the output dir. If you
     rerun with a **changed** model set / K / language / conditions, the
     pipeline detects it and **rebuilds from scratch** (rather than appending
     mismatched rows), reusing whatever in the `work/` cache still matches.
- **Restart.** Just rerun the same command. Books already present in
  `book_to_moral_long.csv` are skipped (logged `SKIP`), and processing picks up
  at the first unfinished book. If the output CSV is missing/partial (e.g. an
  early kill before the first checkpoint), the stage cache still makes the
  re-run cheap — completed stages return instantly with no API calls.
- **`--rebuild`.** Ignores the completed-book skip and rebuilds the whole table
  from the `work/` cache. Use this after changing `--k` or `--labeling-models`
  (it fills only the missing `(moral, model, run)` cells, then re-emits all
  rows). Resumption and rebuild are both idempotent — no duplicate rows.
- **Deterministic.** `temperature=0` everywhere; label-order seeds are derived
  deterministically from `--run-seed`, so the same command reproduces the same
  shuffles (and is saved per row).
- **Fail soft.** Per-book `try/except`; failures are logged to `errors.log` with
  the stage and message, and the run continues.
- **Progress.** One line per stage: `timestamp | book_id | stage | status`.

---

## Implementation notes

- **Chunking.** Stage A splits the full text into fixed token windows
  (`CHUNK_SIZE_TOKENS`, `o200k_base` encoding), summarizes each, and concatenates
  — so it scales to books of any length.
- **Label parsing.** Stage D expects a JSON array of taxonomy labels; the parser
  strips code fences and keeps the **last** `[...]` block in the reply. Models
  occasionally return a label outside the taxonomy — these are kept as-is, not
  silently dropped, so you can audit/filter them downstream.
- **Determinism.** Generation runs at `temperature=0`, and each moral's
  label-order shuffle is seeded deterministically from `--run-seed`, so the same
  command reproduces the same results (the per-row seed is saved in `run_seed`).
- **API endpoints.** OpenAI/Qwen use a JSON-schema-constrained response; Gemini
  uses JSON response mode. Add models by name — the provider is inferred from the
  model prefix (`gpt-`, `gemini-`, `claude-`, `qwen-`) and the matching API key
  is read from the environment.
- **Concurrency.** Labeling runs sequentially behind the global `--throttle-rps`
  rate gate. `--max-active` is reserved for a future parallel labeling path and
  is not yet used to fan out in-flight requests.
