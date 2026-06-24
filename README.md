# CN-GongWen Benchmark（中国公文测试数据生成）

构建两套**中国党政机关公文**评测数据集的可复现工作区，对标
《党政机关公文处理工作条例》(2012) 规定的 **15 种法定公文** 与
GB/T 9704—2012《党政机关公文格式》规定的**格式要素**。面向考察顶级大模型
（Claude / GPT / Gemini）在公文写作、理解、要素抽取、格式合规与办理场景下的能力。

> 🏥 **约一半内容为医疗卫生政策方向**：语料与问题集中约 50% 围绕医疗政策，细分覆盖
> **16 个医疗政策子领域**——医保管理、医药供应与集采、医疗服务价格、公立医院改革、分级诊疗、
> 公共卫生、基层卫生、中医药、药品器械监管、医疗质量与安全、妇幼健康、老龄与医养结合、
> 健康促进、卫生人才与教育、职业健康、互联网医疗与数据。每个子领域再细化为**十分具体的政策主题**
> （如 DRG/DIP 付费、国谈药双通道、院感防控、出生缺陷防治、安宁疗护、互联网诊疗……合计**约 105 个具体分类**）。
> 另含 `policy_domain_classification` **三级分类任务**（政策领域 → 医疗子领域 → 具体政策主题）与
> 医疗专属安全陷阱（患者隐私、把临床指南当强制命令、伪造医保目录条款等）。

> ⚠️ 数据说明：仓库内所有机关名称、姓名、发文字号、公文内容均为**合成示例**，
> 以“示范”机关与 `GA###` 编码匿名化，不对应任何真实单位或真实公文，亦不含个人隐私信息。

## 在 Google Colab 中复现

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-PA/blob/main/notebooks/CN_GongWen_Reproduce_Colab.ipynb)

**一键完美复现**：[`notebooks/CN_GongWen_Reproduce_Colab.ipynb`](notebooks/CN_GongWen_Reproduce_Colab.ipynb)
克隆仓库 → 安装依赖（含中文字体）→（可选）配置 **MiniMax** → 从零重新生成 →
**用 `git diff` 当场证明与提交工件逐字节一致** → 严格校验 → 单元测试 → 公文分布与医疗三级分类可视化 →
证据复核 → 打分器演示。因数据生成完全基于 SHA-256 哈希、**无随机数、核心无第三方依赖**，
任何环境/任何时间复现都逐字节一致。

> 另有 [`notebooks/CN_GongWen_Colab_Pipeline.ipynb`](notebooks/CN_GongWen_Colab_Pipeline.ipynb)
> 作为分步讲解版 pipeline。所有数值/排序/合规标签均由 Python 确定性计算，MiniMax 仅在**事实护栏**
> 下做表层改写，因此有无 LLM 产物完全一致。

## 已实现的四套基准

- **CN-GongWen-Q**：纯自然语言问题集，压力测试文种判定、行文方向、格式要素识别、
  适用情形、常见错误辨析、边界精度、否定枚举、管理开放题、模糊澄清、**18 种幻觉/安全陷阱**
  以及口语化噪声问题（共 11 类问题类型）。
- **CN-GongWen-DataQA**：基于合成公文语料的数据问答，结构化记录 + Python 确定性答案 +
  证据行 + 计算说明 + 异常标签 + 优先级排序 + 接地播报 + 政策领域/医疗子领域分类
  （共 16 类任务，含 7 类进阶难题、5 种播报子类型）。
- **CN-GongWen-Writing**：公文**写作测试 prompt**，按**目标产出 token** 分短/中/长三档，
  对标四类规范（《党政机关公文处理工作条例》+ GB/T 9704 格式 + GB/T 15834 标点 + GB/T 15835 数字），
  蕴含复杂行文框架与行文规则（标题三要素、层次序数、上行文签发人、请示一文一事单一主送、报告不夹带请示、
  函为平行文）、**内容可执行性**（责任—时限—反馈）、**语言安全**（规避夸大/网络化表述）与文种辨析，
  覆盖全部 15 法定文种。测试 prompt 在配置 `MINIMAX_API_KEY` 时由 LLM **一次生成 10 条**、否则确定性模板
  （提交即冻结）；评分 rubric 与参考公文为**确定性事实接地**，故金标准自评满分、逐字节可复现。
  配套 `scorer.py --dataset writing` 做 **11 维**结构化合规打分。
- **CN-GongWen-Audit**：公文**审核/纠错**任务（对标"审核清单/十项硬核自查"）。给一份**确定性注入了
  若干雷区**的公文（含约 1/4 完全合规的对照样本），要求模型找出全部违规；覆盖 **11 类违规**
  （标题缺文种、字号方括号/加"第"、成文日期用汉字、层次序号顿号、夸大用语、编造法规条款、请示多头主送、
  上行文缺签发人/抄送下级、报告夹请示）。金标准由独立检测器校验诚实性，按违规类型 precision/recall/F1 +
  逐项精确匹配 + 合规件零误报打分（`scorer.py --dataset audit`），完全确定性、逐字节可复现。

## 生成数据集

代表性提交档（standard）：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard
```

生产规模本地生成（37 机关、30 个工作日）：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile full --q-count 1000 --dataqa-questions 3000
```

## LiteLLM / MiniMax 配置

LiteLLM 为可选项，仅用于安全的问题改写或播报语言润色。文种、字号、日期、密级、数值、
排序、合规判定始终由 Python 从 `records.csv` 计算。

```bash
export MINIMAX_API_KEY=...
export MINIMAX_API_BASE=https://your-openai-compatible-relay/v1
export MINIMAX_MODEL=MiniMax-M1
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --use-litellm
```

## 校验与评测

```bash
python gongwen_benchmark/scripts/validate_artifacts.py
pytest -q
```

打分器示例（以金标准自评作连通性检查）：

```bash
python gongwen_benchmark/evaluation/scorer.py --dataset q \
  --gold gongwen_benchmark/dataset_1_question_only/questions_with_hidden_metadata.jsonl \
  --pred your_q_predictions.jsonl
python gongwen_benchmark/evaluation/scorer.py --dataset dataqa \
  --gold gongwen_benchmark/dataset_2_data_qa/answers.jsonl \
  --pred your_dataqa_predictions.jsonl
# 写作测试（预测写成 {"question_id": "...", "answer": "<公文全文>"}）
python gongwen_benchmark/evaluation/scorer.py --dataset writing \
  --gold gongwen_benchmark/dataset_3_writing/writing_prompts_with_rubric.jsonl \
  --pred your_writing_predictions.jsonl
# 审核/纠错（预测写成 {"question_id": "...", "violations": ["code", ...]}）
python gongwen_benchmark/evaluation/scorer.py --dataset audit \
  --gold gongwen_benchmark/dataset_4_audit/audit_tasks_with_gold.jsonl \
  --pred your_audit_predictions.jsonl
```

对经批准的脱敏聚合真实台账，使用 `--records-input`。导入器会拒绝个人隐私字段并默认匿名化机关。
完整工作流见 [`gongwen_benchmark/README.md`](gongwen_benchmark/README.md)。

## 二进制工件策略

PR 评审系统不支持二进制 diff，故仓库仅提交可审阅的文本工件（CSV / JSONL）。
需要 Parquet 时本地生成：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py \
  --profile standard --export-parquet /tmp/gongwen-records.parquet
```
