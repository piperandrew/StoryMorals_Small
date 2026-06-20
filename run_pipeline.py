#!/usr/bin/env python3
"""
run_pipeline.py
==============

One Python-driven orchestrator for the Book-to-Moral cultural-comparison
pipeline. It walks a directory of book .txt files and, per book, builds the
DAG:

    full_text ─┬─────────────────────────────► (condition: full_text)
               │
               └─► chunk_summary ─┬───────────► (condition: chunk_summary)
                                  │
                                  └─► short_summary  (condition: short_summary)

For EACH of the three input conditions:
    • generate 3 candidate morals               (Stage C)
    • label each moral against the taxonomy      (Stage D), K order-runs/model

Then it emits one tidy LONG table with one row per assigned value.

The per-stage logic itself lives in pipeline_stages.py (ported verbatim from
the R scripts / notebook). This file only orchestrates: caching, checkpoint/
resume, fail-soft error handling, progress logging and the final table.

Run `python run_pipeline.py --help` for options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import pipeline_config as C
import pipeline_stages as S


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(book_id: str, stage: str, status: str) -> None:
    print(f"{now_iso()} | {book_id:<28} | {stage:<14} | {status}", flush=True)


def log_error(cfg, book_id: str, stage: str, msg: str) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "errors.log", "a", encoding="utf-8") as f:
        f.write(f"{now_iso()}\t{book_id}\t{stage}\t{msg}\n")


def read_cached_text(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path):
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(text) -> str:
    """Short content fingerprint, for invalidating caches when an upstream
    artifact's text changes (not just when a model name changes)."""
    from hashlib import sha256
    return sha256(("" if text is None else str(text)).encode("utf-8")).hexdigest()[:16]


def _morals_fingerprint(morals_by_model: dict) -> str:
    """Stable fingerprint of all morals feeding a condition's labeling step."""
    flat = [(m, t) for m in sorted(morals_by_model) for t in morals_by_model[m]]
    return _hash(json.dumps(flat, ensure_ascii=False, sort_keys=True))


def culture_of(book_id: str) -> str:
    """Culture/language code = filename prefix before the first underscore."""
    return book_id.split("_", 1)[0] if "_" in book_id else "UNK"


def input_type_for(condition: str) -> str:
    # full_text -> the "fulltext" prompt; the two summaries -> the "summary" prompt.
    return "fulltext" if condition == "full_text" else "summary"


# ---------------------------------------------------------------------------
# Output tables: schema + atomic, per-book checkpointing + restart loading
# ---------------------------------------------------------------------------
LONG_COLUMNS = ["book_id", "culture", "input_condition", "moral_index",
                "moral_text", "value_label", "moral_model", "label_model",
                "run_seed", "timestamp"]
SUMMARY_COLUMNS = ["book_id", "culture", "chunk_summary", "short_summary"]

# Collapsed table: one row per (moral_text x moral_model), with value_label =
# the unique value labels pooled across ALL label models for that moral.
UNIQUE_COLUMNS = ["book_id", "culture", "input_condition", "moral_index",
                  "moral_text", "value_label", "moral_model"]


def collapse_unique(df):
    """Derive the one-row-per-moral table.

    Groups by (book, culture, condition, moral_index, moral_text, moral_model)
    and concatenates the DISTINCT value_labels assigned by every label model
    into one comma-separated string. Drops label_model / run_seed / timestamp.
    A moral with no labels keeps one row with an empty value_label.
    """
    if df.empty:
        return pd.DataFrame(columns=UNIQUE_COLUMNS)
    # Works on the new schema (moral_model) or the old single-model one (model).
    mm = "moral_model" if "moral_model" in df.columns else "model"
    keys = ["book_id", "culture", "input_condition", "moral_index",
            "moral_text", mm]

    def join_unique(s):
        vals = sorted({str(x).strip() for x in s.dropna() if str(x).strip()})
        return ", ".join(vals) if vals else None

    out = (df.groupby(keys, dropna=False)["value_label"]
             .apply(join_unique).reset_index()
             .rename(columns={mm: "moral_model"}))
    return (out[UNIQUE_COLUMNS]
            .sort_values(["book_id", "input_condition", "moral_model",
                          "moral_index"])
            .reset_index(drop=True))


def _atomic_write(path: Path, write_fn) -> None:
    """Write via a .tmp sibling then os.replace, so a kill mid-write can never
    corrupt the existing checkpoint (replace is atomic on the same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def write_outputs(cfg, all_rows, all_summaries):
    """(Re)write the long table (CSV+parquet) and the summaries CSV atomically."""
    df = pd.DataFrame(all_rows, columns=LONG_COLUMNS)
    summ_df = pd.DataFrame(all_summaries, columns=SUMMARY_COLUMNS)
    _atomic_write(cfg.output_dir / "book_to_moral_long.csv",
                  lambda p: df.to_csv(p, index=False))
    _atomic_write(cfg.output_dir / "book_summaries.csv",
                  lambda p: summ_df.to_csv(p, index=False))
    # Collapsed one-row-per-moral table (unique labels across label models).
    uniq_df = collapse_unique(df)
    _atomic_write(cfg.output_dir / "book_to_moral_unique.csv",
                  lambda p: uniq_df.to_csv(p, index=False))
    parquet_status = "ok"
    try:
        _atomic_write(cfg.output_dir / "book_to_moral_long.parquet",
                      lambda p: df.to_parquet(p, index=False))
        _atomic_write(cfg.output_dir / "book_to_moral_unique.parquet",
                      lambda p: uniq_df.to_parquet(p, index=False))
    except Exception as e:
        parquet_status = f"SKIPPED (no parquet engine: {e})"
    return df, summ_df, parquet_status


def load_existing_outputs(cfg):
    """Seed accumulators from a prior run so a restart appends instead of
    recomputing. Returns (rows, summaries, done_book_ids)."""
    rows, summaries, done = [], [], set()
    long_path = cfg.output_dir / "book_to_moral_long.csv"
    summ_path = cfg.output_dir / "book_summaries.csv"
    if long_path.exists() and long_path.stat().st_size > 0:
        ldf = pd.read_csv(long_path, dtype={"book_id": str})
        ldf = ldf.where(pd.notnull(ldf), None)   # NaN -> None (NA labels etc.)
        rows = ldf.to_dict("records")
        done = set(ldf["book_id"].astype(str))
    if summ_path.exists() and summ_path.stat().st_size > 0:
        sdf = pd.read_csv(summ_path, dtype={"book_id": str})
        sdf = sdf.where(pd.notnull(sdf), None)
        summaries = [r for r in sdf.to_dict("records")
                     if str(r["book_id"]) in done]   # keep only completed books
    return rows, summaries, done


# ---------------------------------------------------------------------------
# Cached stage wrappers (skip work whose artifact already exists & is non-empty)
# ---------------------------------------------------------------------------
# A per-book manifest records which model produced each single-model artifact
# (the summaries), so changing the chunk/short model invalidates stale caches
# instead of silently reusing them.
def _manifest_path(book_dir):
    return book_dir / "stage_models.json"


def get_chunk_summary(full_text, book_dir, book_id, cfg) -> str:
    path = book_dir / "chunk_summary.txt"
    man = read_json(_manifest_path(book_dir)) or {}
    cached = read_cached_text(path)
    if cached is not None and man.get("chunk_model") == cfg.chunk_model:
        log(book_id, "chunk_summary", "cached")
        return cached
    log(book_id, "chunk_summary", f"running ({cfg.chunk_model})")
    summary = S.stage_chunk_summary(full_text, cfg)
    write_text(path, summary)
    man["chunk_model"] = cfg.chunk_model
    write_json(_manifest_path(book_dir), man)
    log(book_id, "chunk_summary", "done")
    return summary


def get_short_summary(chunk_summary, book_dir, book_id, cfg) -> str:
    path = book_dir / "short_summary.txt"
    man = read_json(_manifest_path(book_dir)) or {}
    chunk_hash = _hash(chunk_summary)
    cached = read_cached_text(path)
    fresh = (cached is not None and man.get("short_model") == cfg.short_model
             and man.get("short_input_hash") == chunk_hash)
    if fresh:
        log(book_id, "short_summary", "cached")
        return cached
    log(book_id, "short_summary", f"running ({cfg.short_model})")
    summary = S.stage_short_summary(chunk_summary, cfg)
    write_text(path, summary)
    man["short_model"] = cfg.short_model
    man["short_input_hash"] = chunk_hash
    write_json(_manifest_path(book_dir), man)
    log(book_id, "short_summary", "done")
    return summary


def get_morals(input_text, condition, language, book_dir, book_id, cfg) -> dict:
    """Generate 3 morals per moral_model. Returns {moral_model: [morals]}.

    Cache is keyed by moral_model, so adding a model fills only the new one.
    """
    path = book_dir / f"morals_{condition}.json"
    cached = read_json(path) or {}
    input_hash = _hash(input_text)
    # If the upstream input text or the language changed, all cached morals for
    # this condition are stale -> drop them and regenerate.
    if (cached.get("input_hash") != input_hash
            or cached.get("language") != language):
        by_model = {}
    else:
        by_model = cached.get("morals", {}) if isinstance(cached, dict) else {}

    def _save():
        write_json(path, {"condition": condition,
                          "input_type": input_type_for(condition),
                          "language": language, "input_hash": input_hash,
                          "morals": by_model})

    changed = False
    for model in cfg.moral_models:
        if by_model.get(model, {}).get("pithy_morals"):
            continue
        log(book_id, f"morals[{condition}/{model}]", "running")
        morals = S.stage_generate_morals(input_text, input_type_for(condition),
                                         language, model, cfg)
        by_model[model] = {"timestamp": now_iso(), "pithy_morals": morals}
        changed = True
        _save()
        log(book_id, f"morals[{condition}/{model}]", f"done ({len(morals)})")
    if not changed:
        log(book_id, f"morals[{condition}]", "cached")
    else:
        _save()
    return {m: by_model.get(m, {}).get("pithy_morals", []) for m in cfg.moral_models}


def get_values(morals_by_model, condition, label_list, book_dir, book_id, cfg) -> list[dict]:
    """Full cross: label every (moral_model, moral) with every label_model, K runs.

    Records: {moral_model, moral_index, label_model, run, run_seed, labels, timestamp}.
    Only missing cells are filled, so adding K / models is an incremental resume.
    """
    path = book_dir / f"values_{condition}.json"
    cached = read_json(path) or {}
    morals_hash = _morals_fingerprint(morals_by_model)
    # If any moral text changed, the cached labels are stale -> regenerate all.
    if cached.get("morals_hash") != morals_hash:
        records = []
    else:
        records = cached.get("records", []) if isinstance(cached, dict) else []
    have = {(r["moral_model"], r["moral_index"], r["label_model"], r["run"])
            for r in records}

    needed = []
    for mm in cfg.moral_models:
        n = len(morals_by_model.get(mm, []))
        for mi in range(1, n + 1):
            for lm in cfg.labeling_models:
                for run in range(cfg.k):
                    if (mm, mi, lm, run) not in have:
                        needed.append((mm, mi, lm, run))

    if not needed:
        log(book_id, f"values[{condition}]", "cached")
        return records

    log(book_id, f"values[{condition}]", f"running ({len(needed)} cells)")
    for (mm, mi, lm, run) in needed:
        seed = S.derive_seed(cfg, book_id, condition, mm, mi, run)
        labels = S.stage_label_moral(morals_by_model[mm][mi - 1], label_list,
                                     seed, lm, cfg)
        records.append({
            "moral_model": mm,
            "moral_index": mi,
            "label_model": lm,
            "run": run,
            "run_seed": seed,
            "labels": labels,                 # list[str] or None on failure
            "timestamp": now_iso(),
        })
        # Persist after each cell so a crash mid-condition still resumes cleanly.
        write_json(path, {"condition": condition, "k": cfg.k,
                          "moral_models": cfg.moral_models,
                          "label_models": cfg.labeling_models,
                          "morals_hash": morals_hash, "records": records})
    log(book_id, f"values[{condition}]", "done")
    return records


# ---------------------------------------------------------------------------
# Per-book DAG -> long rows
# ---------------------------------------------------------------------------
def rows_from_records(book_id, culture, condition, morals_by_model, records) -> list[dict]:
    """Expand label records into tidy long rows (one row per assigned value;
    one NA row when a (moral_model, moral, label_model, run) produced zero labels)."""
    rows = []
    for r in records:
        mm = r["moral_model"]
        mi = r["moral_index"]
        morals = morals_by_model.get(mm, [])
        moral_text = morals[mi - 1] if mi - 1 < len(morals) else None
        labels = r.get("labels") or []
        base = {
            "book_id": book_id,
            "culture": culture,
            "input_condition": condition,
            "moral_index": mi,
            "moral_text": moral_text,
            "moral_model": mm,
            "label_model": r["label_model"],
            "run_seed": r["run_seed"],
            "timestamp": r["timestamp"],
        }
        if labels:
            for lab in labels:
                rows.append({**base, "value_label": lab})
        else:
            rows.append({**base, "value_label": None})  # NA, not dropped
    return rows


def process_book(book_path: Path, cfg, label_list):
    """Return (long_rows, summary_record) for one book."""
    book_id = book_path.stem
    culture = culture_of(book_id)
    book_dir = cfg.work_dir / culture / book_id
    rows = []

    full_text = S.read_text_file(book_path)
    chunk_summary = get_chunk_summary(full_text, book_dir, book_id, cfg)
    short_summary = get_short_summary(chunk_summary, book_dir, book_id, cfg)
    summary_record = {
        "book_id": book_id,
        "culture": culture,
        "chunk_summary": chunk_summary,
        "short_summary": short_summary,
    }

    condition_input = {
        "full_text": full_text,
        "chunk_summary": chunk_summary,
        "short_summary": short_summary,
    }
    language = cfg.language_for_culture(culture)

    for condition in cfg.conditions:
        morals_by_model = get_morals(condition_input[condition], condition,
                                     language, book_dir, book_id, cfg)
        if not any(morals_by_model.values()):
            log_error(cfg, book_id, f"morals[{condition}]", "no morals parsed")
            log(book_id, f"morals[{condition}]", "SKIP (empty)")
            continue
        records = get_values(morals_by_model, condition, label_list,
                             book_dir, book_id, cfg)
        rows.extend(rows_from_records(book_id, culture, condition,
                                      morals_by_model, records))

    return rows, summary_record


# ---------------------------------------------------------------------------
# Discovery + driver
# ---------------------------------------------------------------------------
def discover_books(cfg, books_filter=None, culture_filter=None, limit=None):
    files = sorted(cfg.input_dir.glob("*.txt"))
    if culture_filter:
        files = [f for f in files if culture_of(f.stem) in culture_filter]
    if books_filter:
        wanted = set(books_filter)
        files = [f for f in files if f.stem in wanted or f.name in wanted]
    if limit:
        files = files[:limit]
    return files


def run_signature(cfg) -> dict:
    """Identity of this run's configuration. A change invalidates the output
    skip (the rows would mix schemas / models)."""
    return {
        "chunk_model": cfg.chunk_model, "short_model": cfg.short_model,
        "moral_models": list(cfg.moral_models),
        "labeling_models": list(cfg.labeling_models),
        "k": cfg.k, "language": cfg.language,
        "conditions": list(cfg.conditions), "run_seed": cfg.run_seed,
    }


def run(cfg, books_filter=None, culture_filter=None, limit=None):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    label_list = S.load_label_list(cfg)
    books = discover_books(cfg, books_filter, culture_filter, limit)

    # Detect a configuration change vs the previous output in this dir. If the
    # config differs (or the prior output predates run_config.json), appending
    # would mix models/schemas, so we rebuild from scratch instead of skipping.
    sig = run_signature(cfg)
    cfg_path = cfg.output_dir / "run_config.json"
    prev = read_json(cfg_path)
    existing_output = (cfg.output_dir / "book_to_moral_long.csv").exists()
    config_changed = existing_output and prev != sig
    fresh = cfg.rebuild or config_changed

    if fresh:
        all_rows, all_summaries, done = [], [], set()
    else:
        all_rows, all_summaries, done = load_existing_outputs(cfg)
    write_json(cfg_path, sig)

    print(f"\n{'='*70}")
    print(f"Book-to-Moral pipeline | K={cfg.k} | language={cfg.language} | "
          f"mock={cfg.mock}")
    print(f"  chunk/short : {cfg.chunk_model} / {cfg.short_model}")
    print(f"  morals      : {cfg.moral_models}")
    print(f"  labels      : {cfg.labeling_models}  (full cross)")
    print(f"Input: {cfg.input_dir}  ({len(books)} books selected)")
    print(f"Taxonomy: {len(label_list)} labels | conditions={list(cfg.conditions)}")
    if config_changed and not cfg.rebuild:
        print("** Run config changed vs existing output -> REBUILDING from "
              "scratch (previous book_to_moral_long.csv will be overwritten; "
              "model-matching work/ cache is reused). **")
    elif done:
        print(f"Resuming: {len(done)} books already in output -> will skip "
              f"({'rebuild mode' if cfg.rebuild else 'restart'}).")
    print(f"{'='*70}\n")

    n_ok = n_err = n_skip = 0
    parquet_status = "ok"
    for i, book_path in enumerate(books, 1):
        book_id = book_path.stem
        if book_id in done:
            n_skip += 1
            log(book_id, "SKIP", f"{i}/{len(books)} already in output")
            continue
        log(book_id, "START", f"{i}/{len(books)}")
        try:
            rows, summary_record = process_book(book_path, cfg, label_list)
            all_rows.extend(rows)
            all_summaries.append(summary_record)
            done.add(book_id)
            n_ok += 1
            # Checkpoint the output tables after EVERY book, atomically.
            _, _, parquet_status = write_outputs(cfg, all_rows, all_summaries)
            log(book_id, "DONE", f"{len(rows)} rows (checkpointed)")
        except Exception as e:
            n_err += 1
            tb = traceback.format_exc(limit=3)
            log_error(cfg, book_id, "book", f"{e} :: {tb}")
            log(book_id, "ERROR", str(e)[:120])
            continue

    # Final write (covers the all-skipped / zero-new-books case too).
    df, summ_df, parquet_status = write_outputs(cfg, all_rows, all_summaries)

    print(f"\n{'='*70}")
    print(f"Finished: {n_ok} new, {n_skip} skipped, {n_err} errored. "
          f"{len(df)} total rows.")
    print(f"Wrote: {cfg.output_dir / 'book_to_moral_long.csv'}")
    print(f"Parquet: {parquet_status}")
    print(f"Summaries: {cfg.output_dir / 'book_summaries.csv'} ({len(summ_df)} books)")
    if n_err:
        print(f"Errors logged to: {cfg.output_dir / 'errors.log'}")
    print(f"{'='*70}\n")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Book-to-Moral unified pipeline")
    p.add_argument("--input-dir", default=str(C.INPUT_DIR))
    p.add_argument("--work-dir", default=str(C.WORK_DIR))
    p.add_argument("--output-dir", default=str(C.OUTPUT_DIR))
    p.add_argument("--taxonomy", default=str(C.TAXONOMY_CSV))
    p.add_argument("--model", default=None,
                   help="global override: force ONE model on every stage "
                        "(otherwise per-stage defaults below apply)")
    p.add_argument("--chunk-model", default=None,
                   help=f"Stage A model (default: {C.DEFAULT_CHUNK_MODEL})")
    p.add_argument("--short-model", default=None,
                   help=f"Stage B model (default: {C.DEFAULT_SHORT_MODEL})")
    p.add_argument("--moral-models", default=None,
                   help="comma-separated models for Stage C "
                        f"(default: {','.join(C.DEFAULT_MORAL_MODELS)})")
    p.add_argument("--labeling-models", default=None,
                   help="comma-separated models for Stage D "
                        f"(default: {','.join(C.DEFAULT_LABELING_MODELS)})")
    p.add_argument("--k", type=int, default=1,
                   help="label-order randomizations per moral")
    p.add_argument("--language", default="English",
                   help="moral output language, or '__native__' for per-culture")
    p.add_argument("--run-seed", type=int, default=20240601)
    p.add_argument("--throttle-rps", type=float, default=5.0)
    p.add_argument("--max-active", type=int, default=8)
    p.add_argument("--drop-temperature", action="store_true",
                   help="omit the temperature field (gpt-5.x thinking models)")
    p.add_argument("--conditions", default=",".join(C.CONDITIONS))
    p.add_argument("--books", default=None,
                   help="comma-separated book stems/filenames to run")
    p.add_argument("--culture", default=None,
                   help="comma-separated culture codes to include (e.g. DE,JP)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N books (handy for the single-book test)")
    p.add_argument("--mock", action="store_true",
                   help="stub all API calls with deterministic output (no keys)")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore the completed-book skip and rebuild the whole "
                        "table from the work/ cache (use after changing K/models)")
    return p.parse_args(argv)


def _csv_list(s):
    return [x.strip() for x in s.split(",")] if s else []


def cfg_from_args(a) -> C.Config:
    return C.Config(
        model=a.model,
        chunk_model=a.chunk_model,
        short_model=a.short_model,
        moral_models=_csv_list(a.moral_models),
        labeling_models=_csv_list(a.labeling_models),
        k=a.k,
        language=a.language,
        run_seed=a.run_seed,
        throttle_rps=a.throttle_rps,
        max_active=a.max_active,
        drop_temperature=a.drop_temperature,
        input_dir=Path(a.input_dir),
        work_dir=Path(a.work_dir),
        output_dir=Path(a.output_dir),
        taxonomy_csv=Path(a.taxonomy),
        conditions=tuple(c.strip() for c in a.conditions.split(",")),
        mock=a.mock,
        rebuild=a.rebuild,
    )


def main(argv=None):
    a = parse_args(argv)
    cfg = cfg_from_args(a)
    books_filter = [b.strip() for b in a.books.split(",")] if a.books else None
    culture_filter = ([c.strip() for c in a.culture.split(",")]
                      if a.culture else None)
    run(cfg, books_filter=books_filter, culture_filter=culture_filter,
        limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
