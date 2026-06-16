# CN-GongWen 评测报告

- 被测模型：`<model_id>`
- 评测日期：`<date>`
- 数据版本：profile=`standard`，CN-GongWen-Q=`<n>`，CN-GongWen-DataQA=`<n>`

## 1. CN-GongWen-Q（公文写作/理解）

| 指标 | 数值 |
|---|---:|
| contract_validity_rate | |
| doc_type_routing_accuracy（文种判定） | |
| query_type_accuracy（意图类型） | |
| slot_f1（要素槽位） | |
| noisy_query_robustness（口语化稳健） | |
| clarification_accuracy（按需澄清） | |
| safe_refusal_accuracy（按需拒答） | |
| hallucination_resistance_rate（抗幻觉/抗注入） | |

## 2. CN-GongWen-DataQA（基于语料的数据问答）

| 指标 | 数值 |
|---|---:|
| answer_value_accuracy | |
| evidence_accuracy | |
| top5_ranking_accuracy | |
| ndcg_at_5 | |
| anomaly_precision / recall / f1 | |
| briefing_factual_consistency | |
| briefing_hallucination_rate | |

## 3. 难度分层得分（hard 2x / medium 1x / easy 0.5x）

| 难度 | 题量 | 加权得分 |
|---|---:|---:|

## 4. 安全与合规观察

- 误拒可答问题数：
- 接受伪造字号/越级/虚构密级的次数：
- 播报中凭空捏造事实条数：

## 5. 结论与改进建议
