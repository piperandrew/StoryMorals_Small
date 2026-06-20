# Example / demo

A self-contained, copyright-free demo so you can see the pipeline's inputs and
outputs without the real corpus.

- **`EX_TheLamplighter.txt`** — an original, synthetic short story (public
  domain; written for this repo). `book_id` = `EX_TheLamplighter`.
- **`sample_output/`** — **illustrative** output tables for that story, in the
  exact schema the pipeline produces (`book_to_moral_long.csv`,
  `book_to_moral_unique.csv`, `book_summaries.csv`). The morals and value labels
  here were **hand-authored for documentation** — they are not a live model run.

## Reproduce the *structure* yourself (no API keys, no cost)

```bash
python3 run_pipeline.py --mock --input-dir example \
    --output-dir example/demo --work-dir example/demo_work
# writes to example/demo_mock/ with deterministic stub content
```

## Run it for real (needs API keys)

```bash
export OPENAI_API_KEY=...   # and GEMINI_API_KEY for the default config
python3 run_pipeline.py --input-dir example \
    --output-dir example/demo --work-dir example/demo_work
```
