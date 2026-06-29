# CN-GongWen Benchmark (Chinese Official-Document Generation + Model Evaluation)

[中文](README.md) | **English**

A reproducible workspace that builds **four benchmarks of Chinese Party-and-government official
documents (公文)** *and* ships a **multi-model evaluation / leaderboard** toolchain. Aligned with the
**15 statutory document types** of the *Regulations on the Handling of Official Documents of Party and
Government Organs* (2012) and the **format elements** of **GB/T 9704—2012**, and further incorporating
punctuation (GB/T 15834), number usage (GB/T 15835) and dedicated healthcare regulations. It
stress-tests frontier LLMs (Claude / GPT / Gemini / Qwen / DeepSeek …) on **writing, comprehension,
element extraction, format compliance, handling, and audit/correction** of official documents.

This repo does two things:

1. **Benchmark generation** — one command deterministically builds all four datasets (with gold),
   optionally using an LLM to rewrite surface text under fact guards.
2. **Model evaluation** — `eval_runner.py` runs any model from **Poe / Azure / LiteLLM
   (OpenAI/Anthropic/Qwen/DeepSeek…)** across all four datasets, scores it, and compiles a
   **leaderboard with 95% bootstrap confidence intervals**.

> 🏥 **About half the content is healthcare-policy oriented**: covering **16 medical sub-areas ×
> ~105 specific policy topics** (DRG/DIP payment, dual-channel for nationally-negotiated drugs,
> infection control, birth-defect prevention, hospice care, internet diagnosis…), with a
> **three-level policy classification** (policy domain → medical sub-area → specific topic),
> **medical-compliance discrimination questions**, medical writing-compliance rules, and
> **medical-specific audit violations**, aligned with 2024–2026 updates.

> ⚠️ All agency names, personal names, document serial numbers, and document content are
> **synthetic examples**, anonymized with "示范" (Demo) agencies and `GA###` codes; they correspond
> to no real organization or document and contain no personal-privacy information.

## One-click reproduce + evaluate in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-PA/blob/main/notebooks/CN_GongWen_Reproduce_Colab.ipynb)

**One-click perfect reproduction**: [`notebooks/CN_GongWen_Reproduce_Colab.ipynb`](notebooks/CN_GongWen_Reproduce_Colab.ipynb)
runs end-to-end: configure → clone → install deps (incl. CJK fonts) → (optional) multi-provider LLM
rewrite-and-freeze → **one command generates all four datasets** → **`git diff` proves byte-for-byte
identity** → strict validation → unit tests → visualizations → **five scorers' gold self-eval** →
(optional) **run a small real-model evaluation via `eval_runner.py` and produce a leaderboard**.

> Because generation is entirely `SHA-256`-based with **no randomness and no third-party core
> dependency**, regenerating from source in any environment at any time is **byte-for-byte identical**.

## The four benchmarks

| Dataset | Content | Scorer |
|---|---|---|
| **CN-GongWen-Q** | pure-question stress test, **15 question types** | `--dataset q` |
| **CN-GongWen-DataQA** | data QA over a synthetic corpus, **16 task types** + medical 3-level classification | `--dataset dataqa` |
| **CN-GongWen-Writing** | token-bucketed **writing test prompts**, 11-dimension structured scoring | `--dataset writing` |
| **CN-GongWen-Audit** | official-document **audit (find errors)** + **error-correction (rewrite)** | `--dataset audit` / `rewrite` |

- **CN-GongWen-Q**: document-type selection, writing direction, format-element recognition,
  applicability, boundary precision, negative enumeration, open management questions, ambiguity
  clarification, **18 hallucination/safety traps**, spoken/noisy input, plus four explicit
  discrimination types (doc-type misuse, addressing relations, authority boundaries, medical
  compliance) — 15 types total. Strict split isolation: `questions_public.jsonl` +
  `questions_with_hidden_metadata.jsonl`.
- **CN-GongWen-DataQA**: structured records + Python-deterministic answers + evidence rows +
  calculation notes + anomaly labels + priority ranking + grounded briefings + policy/medical
  classification (16 task types, incl. 7 advanced and 5 briefing subtypes).
- **CN-GongWen-Writing**: bucketed by **target output tokens** into short (≤300) / medium (300–800) /
  long (1200–2500), covering all 15 types, embedding complex framing & rules, **executability**,
  **punctuation/number norms**, and **language safety**. Rubrics and reference answers are
  **deterministically fact-grounded** (gold self-eval is perfect). **Medical** items add
  medical-compliance rules and 2024–2026 authorities.
- **CN-GongWen-Audit**: injects a deterministic subset of violations into a deterministic "correct
  draft" (~1/4 are fully-compliant controls), covering **16 violation types** — 11 general + **5
  medical-specific** (efficacy overclaim, patient-privacy leak, research-as-clinical,
  AI-replaces-physician, insurance-fund violation). Gold is honesty-checked by an independent
  detector. Two sub-tasks: **find** `--dataset audit` and **rewrite** `--dataset rewrite`.

Full dataset design: [`gongwen_benchmark/README.md`](gongwen_benchmark/README.md).

## Install

The core generation/validation/scoring needs **only the standard library**. Install optional extras
for model evaluation and LLM rewriting as needed:

```bash
pip install '.[llm]'            # LiteLLM (OpenAI / Anthropic / Qwen / DeepSeek / Together …)
pip install '.[azure]'         # Azure OpenAI (routed via LiteLLM's azure/ prefix)
pip install '.[poe]'           # Poe API (fastapi-poe)
pip install '.[all-providers]' # all of the above
pip install '.[eval]'          # leaderboard bundle: litellm + fastapi-poe + PyYAML (needed by --models)
pip install '.[dev]'           # pytest + PyYAML (run tests)
```

---

# 1. Generate the datasets

Representative committed profile (standard: 37 agencies, 8 working days; Q 600 / DataQA 1000 /
corpus 4725 / Writing 90 / Audit 90):

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard
```

| Profile | Agencies | Working days | Use |
|---|---:|---:|---|
| `mini` | 5 | 2 | quick smoke test |
| `standard` | 37 | 8 | representative committed benchmark |
| `full` | 37 | 30 | production-scale local generation |

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile full --q-count 1000 --dataqa-questions 3000
```

## LLM rewrite-and-"freeze" (optional; surface text of three datasets)

`--use-llm` rewrites only **surface text** under **fact guards** — the **question wording** of
CN-GongWen-Q, the **briefing language** of CN-GongWen-DataQA, and the **test prompts** of
CN-GongWen-Writing. Document types, serial numbers, dates, security levels, figures, ordering,
compliance judgments, plus writing rubrics / reference answers / audit gold are **always computed
deterministically by Python**. LLM calls are disk-cached and retried, and **fall back to the
deterministic template on any single failure**; enabling it without a key **fails fast**. After a
run, `git add` "**freezes**" those artifacts; byte-identical reproduction without `--use-llm` still
uses the deterministic baseline.

Pick any one provider (`--use-litellm` is a backward-compatible alias):

```bash
# LiteLLM (OpenAI / Anthropic / Qwen / DeepSeek / Minimax relay …)
export LLM_API_KEY=sk-...
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider litellm --llm-model openai/gpt-4o-mini

# Azure OpenAI
export AZURE_API_KEY=...  AZURE_API_BASE=https://<resource>.openai.azure.com/
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider azure --llm-model <your-deployment-name>

# Poe
export POE_API_KEY=...
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider poe --llm-model GPT-4o-Mini
```

The writing-prompt generator takes the same flags:

```bash
python gongwen_benchmark/scripts/generate_writing_prompts.py --count 90 \
  --use-llm --provider poe --llm-model Claude-3-7-Sonnet
```

---

# 2. Evaluate your model

`eval_runner.py` is an end-to-end **multi-model evaluator**: it runs any model across all four
datasets → saves per-question predictions → invokes the scorers → compiles a **leaderboard with 95%
bootstrap confidence intervals**. It supports **Poe / Azure / LiteLLM** backends, **resumes** safely
(each prediction is flushed to disk; re-running skips already-answered questions), and accepts
`--limit N` for a cheap smoke test.

```bash
pip install '.[eval]'
```

## Single-model evaluation

```bash
# LiteLLM: OpenAI / Anthropic / Qwen / DeepSeek …
export OPENAI_API_KEY=sk-...
python gongwen_benchmark/scripts/eval_runner.py \
  --provider litellm --model openai/gpt-4o \
  --dataset all --output-dir results/gpt4o

# Azure OpenAI
export AZURE_API_KEY=...  AZURE_API_BASE=https://<resource>.openai.azure.com/
python gongwen_benchmark/scripts/eval_runner.py \
  --provider azure --model <deployment> --api-version 2024-08-01-preview \
  --dataset all --output-dir results/azure-gpt4o

# Poe
export POE_API_KEY=...
python gongwen_benchmark/scripts/eval_runner.py \
  --provider poe --model Claude-3-7-Sonnet \
  --dataset q --output-dir results/claude-poe

# Smoke test (only the first 5 items per dataset — cheaply verify the wiring)
python gongwen_benchmark/scripts/eval_runner.py \
  --provider litellm --model openai/gpt-4o-mini --limit 5 \
  --dataset all --output-dir results/smoke
```

`--dataset` ∈ `q` / `dataqa` / `writing` / `audit` / `rewrite` / `all` (default `all`). Each model's
predictions and `model_meta.json` go to `results/<model-name>/`.

## Multi-model leaderboard (E7 Leaderboard)

List models in YAML (with strong/medium/weak tiers and all three providers; see
[`models_example.yaml`](gongwen_benchmark/scripts/models_example.yaml)):

```yaml
- name: GPT-4o
  provider: litellm
  model: openai/gpt-4o
- name: Azure-GPT-4o
  provider: azure
  deployment: gpt-4o
  api_base: https://myresource.openai.azure.com/
- name: Claude-via-Poe
  provider: poe
  bot_name: Claude-3-7-Sonnet
```

```bash
# Evaluate every model in sequence, then auto-compile results/leaderboard.json
python gongwen_benchmark/scripts/eval_runner.py \
  --models gongwen_benchmark/scripts/models_example.yaml \
  --dataset all --output-dir results/

# Re-compile the leaderboard from saved predictions only (no model calls)
python gongwen_benchmark/scripts/eval_runner.py --leaderboard-only --output-dir results/
```

The leaderboard ranks by `macro_avg` (the unweighted mean of all leaf metric scores across the four
datasets), with a 95% CI from 1000 bootstrap resamples of model scores.

## Per-provider environment variables

| Provider | Required | Optional / aliases |
|---|---|---|
| **litellm** | `LLM_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` …) | `LLM_MODEL`, `LLM_API_BASE` (for a self-hosted relay) |
| **azure** | `AZURE_API_KEY`, `AZURE_API_BASE` | `AZURE_DEPLOYMENT`, `AZURE_API_VERSION` (default `2024-08-01-preview`) |
| **poe** | `POE_API_KEY` ([poe.com/api_key](https://poe.com/api_key)) | `POE_BOT_NAME` |

Common: `CN_GW_LLM_TEMPERATURE`(0.35), `CN_GW_LLM_TIMEOUT`, `CN_GW_LLM_RETRIES`(3),
`CN_GW_LLM_CACHE`(cache dir), `CN_GW_EVAL_DELAY`(seconds between calls), `CN_GW_BOOTSTRAP_N`(1000),
`CN_GW_CONTEXT_ROW_CAP`(DataQA context row cap, 0=unlimited; set a bound for small-context models),
`CN_GW_EVAL_PREFLIGHT`(default 1; set 0 to skip the one-call startup probe).

> **Reliability**: evaluation runs a **provider preflight** (library/key/endpoint + one probe
> call) before any dataset; on failure it exits non-zero and skips that model — a broken
> environment cannot slip onto the leaderboard as a silent ~0.57. Predictions are flushed
> per-question; add `--resume` to continue an interrupted run (default is a fresh overwrite).

## Underlying scorers (auto-invoked by eval_runner; usable standalone)

Five scorers, with gold self-eval as a connectivity check (ideal score 1.0):

```bash
# 1) CN-GongWen-Q       pred: {"question_id","target_doc_type","expected_query_type","requires_clarification","should_refuse"}
python gongwen_benchmark/evaluation/scorer.py --dataset q \
  --gold gongwen_benchmark/dataset_1_question_only/questions_with_hidden_metadata.jsonl --pred your_q.jsonl
# 2) CN-GongWen-DataQA  pred: {"question_id","answer_value","evidence_rows"}
python gongwen_benchmark/evaluation/scorer.py --dataset dataqa \
  --gold gongwen_benchmark/dataset_2_data_qa/answers.jsonl --pred your_dataqa.jsonl
# 3) CN-GongWen-Writing pred: {"question_id","answer":"<full official document>"}
python gongwen_benchmark/evaluation/scorer.py --dataset writing \
  --gold gongwen_benchmark/dataset_3_writing/writing_prompts_with_rubric.jsonl --pred your_writing.jsonl
# 4) CN-GongWen-Audit / find    pred: {"question_id","violations":["code",...]}
python gongwen_benchmark/evaluation/scorer.py --dataset audit \
  --gold gongwen_benchmark/dataset_4_audit/audit_tasks_with_gold.jsonl --pred your_audit.jsonl
# 5) CN-GongWen-Audit / rewrite pred: {"question_id","rewrite":"<rewritten official document>"}
python gongwen_benchmark/evaluation/scorer.py --dataset rewrite \
  --gold gongwen_benchmark/dataset_4_audit/audit_tasks_with_gold.jsonl --pred your_rewrite.jsonl
```

---

## Validation & unit tests

```bash
python gongwen_benchmark/scripts/validate_artifacts.py   # strict cross-file validation (4 datasets + gold honesty)
pytest -q
```

The strict validator checks public/hidden isolation, cross-file IDs, evidence integrity, the answer
contract, task/question-type coverage, difficulty distribution (hard ≥40%, easy ≤25%), trap diversity
(≥12), medical share (≈50%), writing reference answers landing in their token buckets, and audit gold
honesty (injected ⇔ detectable; each corrected draft is itself violation-free).

## Approved real / hybrid data

For an approved, de-identified, aggregated ledger, pass `--records-input`; the importer rejects
personal-privacy fields and anonymizes agencies by default. See
[`gongwen_benchmark/README.md`](gongwen_benchmark/README.md).

## Repository layout

```
gongwen_benchmark/
├─ dataset_1_question_only/    # CN-GongWen-Q: public + hidden + taxonomy
├─ dataset_2_data_qa/          # CN-GongWen-DataQA: records.csv + questions/answers/...
├─ dataset_3_writing/          # CN-GongWen-Writing: public prompts + with_rubric (reference docs) + taxonomy
├─ dataset_4_audit/            # CN-GongWen-Audit: public (to audit/rewrite) + with_gold (violations + correct draft) + taxonomy
├─ evaluation/                 # scorer.py (5 scorers) + metrics/rules/report templates
├─ workflow/                   # official-document handling (OA) event-stream examples
└─ scripts/
   ├─ generate_benchmarks.py       # main generator (Q + DataQA + corpus)
   ├─ generate_writing_prompts.py  # writing test-prompt generator
   ├─ generate_audit_tasks.py      # audit / rewrite task generator
   ├─ eval_runner.py               # ★ multi-model evaluator + leaderboard (Poe/Azure/LiteLLM)
   ├─ llm_providers.py             # ★ unified LLM provider abstraction (cache/retry/fact-guard)
   ├─ models_example.yaml          # ★ multi-model evaluation config example
   ├─ litellm_minimax.py           # backward-compat shim (re-exports llm_providers)
   ├─ benchmark_schema.py          # doc-type / element / medical 3-level / trap schemas
   ├─ data_sources.py              # approved real-data import (rejects privacy fields, anonymizes)
   └─ validate_artifacts.py        # strict cross-file validator
```

## Binary-artifact policy

PR review does not support binary diffs, so the repo commits only reviewable text artifacts
(CSV / JSONL). Generate Parquet locally when needed:

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --export-parquet /tmp/gongwen-records.parquet
```
