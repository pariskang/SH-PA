# CN-GongWen Benchmark

本目录包含两套**中国党政机关公文**评测基准的可复现生成器与提交工件，
**面向挑战顶级大模型（Claude / GPT / Gemini）**。提交切分在两套数据集上保持
高难题占比 ≥40%。所有数据均为合成示例，不对应真实单位或真实公文。

## 标准依据

- 《党政机关公文处理工作条例》(2012)：15 种法定公文文种。
- GB/T 9704—2012《党政机关公文格式》：18 个版头/主体/版记格式要素。

## 三套数据集

1. **CN-GongWen-Q**：纯问题自然语言压力测试，覆盖 11 类问题类型：
   - `single_doc_type`、`multi_doc_type`、`cross_element_chain`（跨要素链式推理）、
     `temporal_compound`（时效复合）、`conflicting_signals`（文种—行文方向冲突）、
     `boundary_precision`（份号位数/字号格式等精确规则）、`negative_enumeration`（否定枚举）、
     `management_open`、`ambiguous_boundary`、`hallucination_trap`（**18 种不同陷阱**）、
     `spoken_noisy`。
   - 行文关系与权限雷区已扩充：主送上级领导个人、越级行文未抄送被越过机关、部门越权/未会签擅自向下行文、
     内设机构对外行文、方案/计划/总结/申请等非法定文种误用（分别并入 `conflicting_signals` /
     `management_open` / `negative_enumeration`，不改变题型配比与医疗/难度分布）。
2. **CN-GongWen-DataQA**：基于合成公文语料的数据问答，覆盖 16 类任务：
   - 基线 8：`direct_lookup`、`cross_agency_ranking`、`period_comparison`、`sustained_trend`、
     `composite_element_explanation`、`anomaly_detection`、`priority_ranking`、`briefing`。
   - **7 类进阶难题**：`cross_doc_extremum`、`consecutive_compliance_streak`、
     `counterfactual_format`、`quality_filtered_aggregate`、`negative_enumeration`、
     `multi_criteria_ranking`、`precision_percentage_change`。
   - **政策领域分类**：`policy_domain_classification`（判别通用政务/医疗卫生，并归入 16 个医疗子领域）。
   - **5 种播报子类型**：`standard_executive`、`risk_focused_targeted`、
     `conflicting_signals_briefing`、`exclusion_briefing`、`leadership_focus`。
3. **CN-GongWen-Writing**：公文**写作测试 prompt**，按**目标产出 token** 分桶（short ≤300 /
   medium 300–800 / long 1200–2500），覆盖全部 15 法定文种，蕴含复杂行文框架与硬性行文规则：
   - **权威依据**（schema `STANDARDS`）：《党政机关公文处理工作条例》+ GB/T 9704—2012（格式）+
     GB/T 15834—2011（标点）+ GB/T 15835—2011（数字用法）。
   - 框架维度：标题三要素（机关+事由+文种）、层次序数 `一、（一）1.`、上行文签发人、主送/抄送、
     附件说明、附注（请示注明联系人电话）、结尾用语、署名与成文日期（阿拉伯数字）。
   - 行文规则约束：请示一文一事且单一主送、报告不夹带请示、上行文不抄送下级、函为平行文、
     涉密标份号与密级；并以 18 类陷阱 + 文种典型雷区作为**须规避雷区**的负向约束。
   - **内容可执行性**：部署性下行文（通知/意见/决定）须写明责任主体、完成时限、报送反馈
     （依据—目标—任务—责任—时限—保障—反馈），避免“只有口号没有抓手”。
   - **语言安全**：规避夸大/网络化/绝对化表述（schema `FORBIDDEN_PHRASES`）；**文种辨析**
     （schema `DOC_TYPE_GUIDE`）为每文种附主要用途、写作要点、典型雷区（参考请示/报告/函辨析表）；
     另含**十项硬核自查** `SELF_CHECK`。
   - 生成与复现：`MINIMAX_API_KEY` 就绪时由 LLM **一次 10 条**撰写测试 prompt、否则确定性模板（提交即冻结）；
     评分 `rubric` 与 `reference_answer`（合规且命中目标 token 区间的参考公文）始终确定性、事实接地，
     故金标准自评满分、逐字节可复现。打分见 `scorer.py --dataset writing`，**11 个结构化维度**：
     长度/标题/层次/署名日期/签发人/主送/行文方向/**可执行性/标点/语言安全**/雷区规避。

## 医疗卫生政策方向（约占一半）

语料与问题集中**约 50% 为医疗卫生政策方向**，并细分为 **16 个子领域 × 共约 105 个十分具体的政策主题**
（三级分类：政策领域 → 医疗子领域 → 具体政策主题；完整清单见 `scripts/benchmark_schema.py: MEDICAL_AREAS`
与 `dataset_1_question_only/taxonomy.json`）。下表每个子领域列出其代表性具体主题（节选）：

| 子领域 | 代表性具体政策主题（节选，完整 6~9 个/子领域） |
|---|---|
| 医保管理 | DRG按病种分组付费改革、DIP按病种分值付费改革、医保药品目录管理、医保基金常态化监管、门诊共济保障、异地就医直接结算、长期护理保险、生育保险待遇 |
| 医药供应与集采 | 药品集中带量采购落地、高值耗材带量采购、国谈药“双通道”管理、短缺药保供稳价、基本药物制度、药品价格监测 |
| 医疗服务价格 | 价格动态调整、检查检验价格规范管理、价格项目立项编制、价格备案 |
| 公立医院改革 | 高质量发展、绩效国家监测考核、薪酬制度改革、现代医院管理与章程、全面预算运营、党建引领 |
| 分级诊疗 | 城市医疗集团、县域医共体、家庭医生签约、双向转诊、基层首诊、专科联盟/远程协作网 |
| 公共卫生 | 重大传染病防控、疾控体系改革、免疫规划、突发公卫应急、慢病综合防控、地方病防治、严重精神障碍管理、病媒生物防制 |
| 基层卫生 | 社区卫生服务中心标准化、乡镇卫生院能力提升、村卫生室管理、基本公卫服务项目、家庭医生团队、人才“县管乡用” |
| 中医药 | 传承创新发展、中医机构能力建设、中西医结合协作、中药饮片质量、适宜技术推广、中医馆建设 |
| 药品器械监管 | 药品生产经营监管、器械注册与使用、疫苗全程追溯、药物警戒、麻精药品管理、药品网络销售、化妆品监管 |
| 医疗质量与安全 | 质量管理与控制、院感预防控制、临床用血、合理用药/处方点评、抗菌药物管理、纠纷预防处理、手术安全核查、病案质量 |
| 妇幼健康 | 母婴安全、出生缺陷综合防治、近视防控、普惠托育、“两癌”筛查、婚前孕前保健 |
| 老龄与医养结合 | 医养结合服务、安宁疗护、老年健康体系、失能长期照护、老年友善医疗机构、认知障碍防治 |
| 健康促进 | 健康中国行动、控烟、全民健身与健康融合、国民营养计划、社会心理服务、健康城市、健康素养提升 |
| 卫生人才与教育 | 住院医师规范化培训、全科医生培养、执业注册管理、护士队伍建设、继续教育、职称评审、紧缺人才培养 |
| 职业健康 | 职业病防治、职业健康监护、用人单位主体责任、尘肺病防治、分类监督执法、放射卫生防护 |
| 互联网医疗与数据 | 互联网医院/诊疗、远程医疗协同网、电子健康档案、健康大数据、智慧医院/电子病历分级、结果互认、电子处方流转 |

卫生健康系统机关（卫健委、医保局、疾控局、中医药局、药监局、卫生监督所）发文以医疗为主；
综合机关（政府/办公厅）体现“三医联动”跨部门医疗政策。`policy_domain_classification` 任务要求
模型给出**三级具体分类**（政策领域→医疗子领域→具体政策主题），金标准由 Python 依据语料确定。
医疗方向还含专属安全陷阱：患者隐私索取、把临床诊疗指南当强制命令、伪造医保目录条款、
虚标疾病监测周报密级、越界统计真实疫情数据等。

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
├─ dataset_3_writing/          # CN-GongWen-Writing：public prompt + with_rubric(参考公文) + taxonomy
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
