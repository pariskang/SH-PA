# CN-GongWen Benchmark

本目录包含两套**中国党政机关公文**评测基准的可复现生成器与提交工件，
**面向挑战顶级大模型（Claude / GPT / Gemini）**。提交切分在两套数据集上保持
高难题占比 ≥40%。所有数据均为合成示例，不对应真实单位或真实公文。

## 标准依据

- 《党政机关公文处理工作条例》(2012)：15 种法定公文文种。
- GB/T 9704—2012《党政机关公文格式》：18 个版头/主体/版记格式要素。

## 两套数据集

1. **CN-GongWen-Q**：纯问题自然语言压力测试，覆盖 11 类问题类型：
   - `single_doc_type`、`multi_doc_type`、`cross_element_chain`（跨要素链式推理）、
     `temporal_compound`（时效复合）、`conflicting_signals`（文种—行文方向冲突）、
     `boundary_precision`（份号位数/字号格式等精确规则）、`negative_enumeration`（否定枚举）、
     `management_open`、`ambiguous_boundary`、`hallucination_trap`（**18 种不同陷阱**）、
     `spoken_noisy`。
2. **CN-GongWen-DataQA**：基于合成公文语料的数据问答，覆盖 15 类任务：
   - 基线 8：`direct_lookup`、`cross_agency_ranking`、`period_comparison`、`sustained_trend`、
     `composite_element_explanation`、`anomaly_detection`、`priority_ranking`、`briefing`。
   - **7 类进阶难题**：`cross_doc_extremum`、`consecutive_compliance_streak`、
     `counterfactual_format`、`quality_filtered_aggregate`、`negative_enumeration`、
     `multi_criteria_ranking`、`precision_percentage_change`。
   - **5 种播报子类型**：`standard_executive`、`risk_focused_targeted`、
     `conflicting_signals_briefing`、`exclusion_briefing`、`leadership_focus`。

## 档位

| Profile | 机关数 | 工作日 | 用途 |
|---|---:|---:|---|
| `mini` | 5 | 2 | 快速冒烟测试 |
| `standard` | 37 | 8 | 提交的代表性基准（约 4700 条公文记录） |
| `full` | 37 | 30 | 生产规模本地生成 |

## 目录结构

```
gongwen_benchmark/
├─ dataset_1_question_only/    # CN-GongWen-Q：public + hidden + taxonomy
├─ dataset_2_data_qa/          # CN-GongWen-DataQA：records.csv + questions/answers/...
├─ evaluation/                 # scorer.py + 度量/规则/报告模板
├─ workflow/                   # 公文办理流转（OA）事件流示例
├─ scripts/                    # 模式、生成器、校验器、真实数据导入、LiteLLM
├─ agency_metadata.csv         # 37 个合成机关
└─ element_dictionary.csv      # 18 个格式要素字典
```

## 生成工件

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard
```

完整规模：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile full --q-count 1000 --dataqa-questions 3000
```

## 经 OpenAI 兼容中继的 LiteLLM / MiniMax

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

LiteLLM 仅改写问题或润色播报文字。文种、行文方向、发文字号、密级、数值、排序、
合规判定与证据行始终由 Python 从 `records.csv` 确定性计算。

## 经批准的真实/混合数据工作流

用 `--records-input` 传入经批准的脱敏聚合台账。导入器拒绝个人隐私字段，并默认匿名化机关。

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --records-input /secure/approved_records.csv
```

## 校验与评测

```bash
python gongwen_benchmark/scripts/validate_artifacts.py
pytest -q
```

严格校验器检查 public/hidden 隔离、跨文件 ID、证据完整性、答案契约、任务覆盖、
难度分布（hard ≥40%、easy ≤25%）、陷阱多样性（≥12 种）、播报事实接地与 padding 占比。
LiteLLM 调用带缓存、重试、JSON 校验与事实漂移护栏。
