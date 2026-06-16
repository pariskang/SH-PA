# CN-GongWen Benchmark（中国公文测试数据生成）

构建两套**中国党政机关公文**评测数据集的可复现工作区，对标
《党政机关公文处理工作条例》(2012) 规定的 **15 种法定公文** 与
GB/T 9704—2012《党政机关公文格式》规定的**格式要素**。面向考察顶级大模型
（Claude / GPT / Gemini）在公文写作、理解、要素抽取、格式合规与办理场景下的能力。

> ⚠️ 数据说明：仓库内所有机关名称、姓名、发文字号、公文内容均为**合成示例**，
> 以“示范”机关与 `GA###` 编码匿名化，不对应任何真实单位或真实公文，亦不含个人隐私信息。

## 在 Google Colab 中复现

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-PA/blob/main/notebooks/CN_GongWen_Colab_Pipeline.ipynb)

[`notebooks/CN_GongWen_Colab_Pipeline.ipynb`](notebooks/CN_GongWen_Colab_Pipeline.ipynb) 是端到端的一键复现笔记本：克隆仓库、安装依赖、（可选）配置
**MiniMax API**（OpenAI 兼容）、连通性自检、用 `generate_benchmarks.py --use-litellm`
生成两套数据集，随后严格校验、单元测试、分布可视化、对照 `records.csv` 复核证据行，
并演示打分器。所有数值/排序/合规标签均由 Python 确定性计算，MiniMax 只在**事实护栏**
下做表层改写，因此有无 LLM 产物完全一致。

## 已实现的两套基准

- **CN-GongWen-Q**：纯自然语言问题集，压力测试文种判定、行文方向、格式要素识别、
  适用情形、常见错误辨析、边界精度、否定枚举、管理开放题、模糊澄清、**18 种幻觉/安全陷阱**
  以及口语化噪声问题（共 11 类问题类型）。
- **CN-GongWen-DataQA**：基于合成公文语料的数据问答，结构化记录 + Python 确定性答案 +
  证据行 + 计算说明 + 异常标签 + 优先级排序 + 接地播报（共 15 类任务，含 7 类进阶难题、
  5 种播报子类型）。

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
