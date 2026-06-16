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
2. **CN-GongWen-DataQA**：基于合成公文语料的数据问答，覆盖 16 类任务：
   - 基线 8：`direct_lookup`、`cross_agency_ranking`、`period_comparison`、`sustained_trend`、
     `composite_element_explanation`、`anomaly_detection`、`priority_ranking`、`briefing`。
   - **7 类进阶难题**：`cross_doc_extremum`、`consecutive_compliance_streak`、
     `counterfactual_format`、`quality_filtered_aggregate`、`negative_enumeration`、
     `multi_criteria_ranking`、`precision_percentage_change`。
   - **政策领域分类**：`policy_domain_classification`（判别通用政务/医疗卫生，并归入 16 个医疗子领域）。
   - **5 种播报子类型**：`standard_executive`、`risk_focused_targeted`、
     `conflicting_signals_briefing`、`exclusion_briefing`、`leadership_focus`。

## 医疗卫生政策方向（约占一半）

语料与问题集中**约 50% 为医疗卫生政策方向**，细分覆盖 16 个医疗政策子领域（见
`scripts/benchmark_schema.py: MEDICAL_AREAS`）：

| 子领域 | 代表政策 |
|---|---|
| 医保管理 | DRG/DIP 支付改革、医保目录、基金监管、门诊共济、异地结算、长护险 |
| 医药供应与集采 | 药品/高值耗材带量采购、国谈药落地、双通道、基本药物制度 |
| 医疗服务价格 | 价格动态调整、检查检验价格规范、价格项目立项 |
| 公立医院改革 | 高质量发展、绩效考核、薪酬改革、现代医院管理 |
| 分级诊疗 | 医联体/医共体、家庭医生签约、双向转诊 |
| 公共卫生 | 传染病防控、疾控改革、免疫规划、应急、慢病/地方病 |
| 基层卫生 | 社区卫生、乡镇卫生院、基本公共卫生服务项目 |
| 中医药 | 传承创新、中医机构建设、中西医结合、中药质量 |
| 药品器械监管 | 药品安全、器械注册、疫苗管理、药物警戒 |
| 医疗质量与安全 | 质量管理、院感防控、临床用血、合理用药、纠纷处理 |
| 妇幼健康 | 母婴安全、出生缺陷防治、近视防控、托育 |
| 老龄与医养结合 | 医养结合、安宁疗护、老年健康、失能照护 |
| 健康促进 | 健康中国行动、控烟、全民健身、营养、心理健康 |
| 卫生人才与教育 | 队伍建设、住培、全科医生、继续教育 |
| 职业健康 | 职业病防治、健康监护、分类监督执法 |
| 互联网医疗与数据 | 互联网+医疗、远程医疗、电子健康档案、健康大数据、智慧医院 |

卫生健康系统机关（卫健委、医保局、疾控局、中医药局、药监局、卫生监督所）发文以医疗为主；
综合机关（政府/办公厅）体现“三医联动”跨部门医疗政策。医疗方向还含专属安全陷阱：患者隐私索取、
把临床诊疗指南当强制命令、伪造医保目录条款、虚标疾病监测周报密级、越界统计真实疫情数据等。

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
