# CN-GongWen Benchmark (Chinese Official-Document Test-Data Generation)

[中文](README.md) | **English**

A reproducible workspace that builds **four benchmarks of Chinese Party-and-government official
documents (公文)**, aligned with the **15 statutory document types** of the *Regulations on the
Handling of Official Documents of Party and Government Organs* (2012) and the **format elements** of
**GB/T 9704—2012**, and further incorporating punctuation (GB/T 15834), number usage (GB/T 15835)
and dedicated healthcare regulations. It is designed to stress-test frontier LLMs (Claude / GPT /
Gemini) on **writing, comprehension, element extraction, format compliance, handling, and
audit/correction** of official documents.

> 🏥 **About half the content is healthcare-policy oriented**: covering **16 medical sub-areas ×
> ~105 specific policy topics** (DRG/DIP payment, dual-channel for nationally-negotiated drugs,
> infection control, birth-defect prevention, hospice care, internet diagnosis…), with a
> **three-level policy classification** (policy domain → medical sub-area → specific topic),
> **medical-compliance discrimination questions**, medical writing-compliance rules, and
> **medical-specific audit violations**, aligned with 2024–2026 updates (Infectious Disease
> Prevention Law 2025, IIT Administrative Measures 2024, Medical Insurance Fund Supervision
> Implementation Rules 2026, etc.).

> ⚠️ All agency names, personal names, document serial numbers, and document content are
> **synthetic examples**, anonymized with "示范" (Demo) agencies and `GA###` codes; they correspond
> to no real organization or document and contain no personal-privacy information.

## One-click test-data generation in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-PA/blob/main/notebooks/CN_GongWen_Reproduce_Colab.ipynb)

**One-click perfect reproduction**: [`notebooks/CN_GongWen_Reproduce_Colab.ipynb`](notebooks/CN_GongWen_Reproduce_Colab.ipynb)
runs end-to-end: configure → clone → install deps (incl. CJK fonts) → (optional) MiniMax LLM
rewrite-and-freeze → **one command generates all four datasets** → **`git diff` proves byte-for-byte
identity with committed artifacts** → strict validation → unit tests → visualizations for all four
datasets (Q 15 types / DataQA 16 tasks / medical 3-level / Writing token buckets / Audit 16
violations) → **five scorers' gold self-eval** → sample tour.

> Because generation is entirely `SHA-256`-based with **no randomness and no third-party core
> dependency**, regenerating from source in any environment at any time is **byte-for-byte identical**.

## The four benchmarks

| Dataset | Content | Scorer |
|---|---|---|
| **CN-GongWen-Q** | pure-question stress test, **15 question types** | `--dataset q` |
| **CN-GongWen-DataQA** | data QA over a synthetic corpus, **16 task types** + medical 3-level classification | `--dataset dataqa` |
| **CN-GongWen-Writing** | token-bucketed **writing test prompts**, 11-dimension structured scoring | `--dataset writing` |
| **CN-GongWen-Audit** | official-document **audit (find errors)** + **error-correction (rewrite)** | `--dataset audit` / `rewrite` |

- **CN-GongWen-Q**: covers document-type selection, writing direction, format-element recognition,
  applicability, boundary precision, negative enumeration, open management questions, ambiguity
  clarification, **18 hallucination/safety traps**, spoken/noisy input, plus four explicit
  discrimination types — **document-type misuse, addressing relations, authority boundaries, and
  medical compliance** (15 types total). Medical-compliance questions probe absolute efficacy claims,
  patient privacy, ethics review, AI-replacing-physician, medical-insurance-fund violations, etc.
  Strict split isolation: `questions_public.jsonl` (questions only) +
  `questions_with_hidden_metadata.jsonl` (offline-scoring metadata).
- **CN-GongWen-DataQA**: structured records + Python-deterministic answers + evidence rows +
  calculation notes + anomaly labels + priority ranking + grounded briefings +
  policy-domain/medical-sub-area classification (16 task types, incl. 7 advanced and 5 briefing subtypes).
- **CN-GongWen-Writing**: bucketed by **target output tokens** into short (≤300) / medium (300–800) /
  long (1200–2500), covering all 15 types, embedding complex framing & rules (title three-elements,
  ordinal hierarchy, signatory on upward documents, one-matter/single-recipient 请示, 报告 not embedding
  a 请示, 函 as a parallel document), **executability** (basis—goal—task—responsibility—deadline—
  safeguards—feedback), **punctuation/number norms** (GB/T 15834/15835), and **language safety**.
  Rubrics and reference answers are **deterministically fact-grounded**; `--dataset writing` scores
  **11 dimensions** (gold self-eval is perfect). **Medical** items add medical-compliance rules
  (no absolute efficacy, de-identification, informed consent, ethics review, AI-as-assist not
  replacement, insurance-fund compliance) and 2024–2026 medical authorities, with medical
  hype/advertising red-line words folded into the language-safety dimension.
- **CN-GongWen-Audit**: injects a deterministic subset of violations into a deterministic "correct
  draft" (~1/4 are fully-compliant controls), covering **16 violation types** — 11 general (title
  missing doc type, square-bracket year / "第" in the serial, Chinese-numeral date, ordinal 顿号,
  hype language, fabricated legal article, multi-head 请示, upward document missing signatory /
  cc-to-subordinate, 报告 embedding a 请示) + **5 medical-specific** (efficacy overclaim, patient-privacy
  leak, research-as-clinical, AI-replaces-physician, insurance-fund violation). Gold is honesty-checked
  by an independent detector. Two sub-tasks:
  - **find** `--dataset audit`: violation-type precision/recall/F1 + per-item exact match + zero
    false-alarm on clean documents.
  - **rewrite** `--dataset rewrite`: rewrite the flawed document into a compliant one; gold is the
    pre-injection compliant draft; scored on **violations-removed / key-facts-preserved / format-valid**.

## Generating the datasets (one command → four datasets)

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

## LLM rewrite-and-"freeze" (optional; covers the surface text of three datasets)

`--use-litellm` rewrites only **surface text** under **fact guards** — the **question wording** of
CN-GongWen-Q, the **briefing language** of CN-GongWen-DataQA, and the **test prompts** of
CN-GongWen-Writing; document types, serial numbers, dates, security levels, figures, ordering,
compliance judgments, plus writing rubrics / reference answers / audit gold are **always computed
deterministically by Python**. LLM calls are disk-cached and retried, and **fall back to the
deterministic template on any single failure** (so a freeze run never aborts); enabling
`--use-litellm` without a key **fails fast with a clear error**. After a run, `git add` to commit
"**freezes**" those LLM artifacts; the byte-identical reproduction without `--use-litellm` still uses
the deterministic baseline.

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

## Validation & evaluation

```bash
python gongwen_benchmark/scripts/validate_artifacts.py   # strict cross-file validation (4 datasets + gold honesty)
pytest -q
```

Five scorers (gold self-eval as a connectivity check; the ideal score is 1.0):

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

The strict validator checks public/hidden isolation, cross-file IDs, evidence integrity, the answer
contract, task/question-type coverage, difficulty distribution (hard ≥40%, easy ≤25%), trap diversity
(≥12), medical share (≈50%), writing reference answers landing in their token buckets, and audit gold
honesty (injected ⇔ detectable; each corrected draft is itself violation-free), among others.

## Approved real / hybrid data

For an approved, de-identified, aggregated ledger, pass `--records-input`; the importer rejects
personal-privacy fields and anonymizes agencies by default. See
[`gongwen_benchmark/README.md`](gongwen_benchmark/README.md) for the full workflow.

## Repository layout

```
gongwen_benchmark/
├─ dataset_1_question_only/    # CN-GongWen-Q: public + hidden + taxonomy
├─ dataset_2_data_qa/          # CN-GongWen-DataQA: records.csv + questions/answers/...
├─ dataset_3_writing/          # CN-GongWen-Writing: public prompts + with_rubric (reference docs) + taxonomy
├─ dataset_4_audit/            # CN-GongWen-Audit: public (to audit/rewrite) + with_gold (violations + correct draft) + taxonomy
├─ evaluation/                 # scorer.py (5 scorers) + metrics/rules/report templates
├─ workflow/                   # official-document handling (OA) event-stream examples
├─ scripts/                    # schema, generators (benchmarks/writing/audit), validator, real-data import, tokens, LiteLLM
├─ agency_metadata.csv         # 37 synthetic agencies
└─ element_dictionary.csv      # 18 format-element dictionary
```

## Binary-artifact policy

PR review does not support binary diffs, so the repo commits only reviewable text artifacts
(CSV / JSONL). Generate Parquet locally when needed:

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --export-parquet /tmp/gongwen-records.parquet
```
