# CN-GongWen Benchmark（中国公文测试数据生成 + 模型评测）

**中文** | [English](README.en.md)

可复现地构建**四套中国党政机关公文**评测数据集，并提供**多模型评测/排行榜**工具链：对标
《党政机关公文处理工作条例》(2012) 的 **15 种法定公文** 与 GB/T 9704—2012《党政机关公文格式》的
**格式要素**，融入标点（GB/T 15834）、数字用法（GB/T 15835）与医疗卫生专项法规。面向考察顶级大模型
（Claude / GPT / Gemini / Qwen / DeepSeek …）在公文**写作、理解、要素抽取、格式合规、办理、审核纠错**
等场景下的能力。

本仓库覆盖两件事：

1. **公文试题生成** —— 一条命令确定性生成四套数据集（含金标准），可选用 LLM 在事实护栏下改写表层文本。
2. **公文生成模型测试评估** —— `eval_runner.py` 用 **Poe / Azure / LiteLLM（OpenAI/Anthropic/Qwen/DeepSeek…）**
   任意模型跑完四套数据集，自动打分并汇总成**带 95% 置信区间的排行榜**。

> 🏥 **约一半内容为医疗卫生政策方向**：覆盖 **16 个医疗子领域 × 约 105 个具体政策主题**
> （DRG/DIP 付费、国谈药双通道、院感防控、出生缺陷防治、安宁疗护、互联网诊疗……），含
> 政策领域**三级分类**（政策领域 → 医疗子领域 → 具体主题）、**医疗合规辨析题**、医疗写作合规规则与
> **医疗专属审核违规**，并对齐 2024–2026 新规（传染病防治法 2025、IIT 管理办法 2024、医保基金监管实施细则 2026 等）。

> ⚠️ 所有机关名称、姓名、发文字号、公文内容均为**合成示例**，以"示范"机关与 `GA###` 编码匿名化，
> 不对应任何真实单位或真实公文，亦不含个人隐私信息。

## 在 Google Colab 中一键复现 + 评测

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/SH-PA/blob/main/notebooks/CN_GongWen_Reproduce_Colab.ipynb)

**一键完美复现**：[`notebooks/CN_GongWen_Reproduce_Colab.ipynb`](notebooks/CN_GongWen_Reproduce_Colab.ipynb)
端到端跑通：配置 → 克隆 → 装依赖（含中文字体）→（可选）多 provider LLM 改写冻结 →
**一条命令生成四套数据集** → **用 `git diff` 当场证明逐字节一致** → 严格校验 → 单元测试 →
四套数据集可视化 → **五个打分器金标准自评** →（可选）**用真实模型跑 `eval_runner.py` 小样本评测并出排行榜**。

> 因数据生成完全基于 `SHA-256`、**无随机数、核心无第三方依赖**，任何环境/任何时间从源码重生成都**逐字节一致**。

## 四套基准

| 数据集 | 内容 | 打分器 |
|---|---|---|
| **CN-GongWen-Q** | 纯问题压力测试，**15 类题型** | `--dataset q` |
| **CN-GongWen-DataQA** | 合成语料数据问答，**16 类任务** + 医疗三级分类 | `--dataset dataqa` |
| **CN-GongWen-Writing** | 按 token 分桶的**写作测试 prompt**，11 维结构化打分 | `--dataset writing` |
| **CN-GongWen-Audit** | 公文**审核找错** + **纠错改写** | `--dataset audit` / `rewrite` |

- **CN-GongWen-Q**：覆盖文种判定、行文方向、格式要素识别、适用情形、边界精度、否定枚举、管理开放题、
  模糊澄清、**18 种幻觉/安全陷阱**、口语化噪声，以及**文种误用、行文关系、权限边界、医疗合规**四类显式辨析题
  （共 15 类）。切分严格隔离：`questions_public.jsonl`（仅问题）+ `questions_with_hidden_metadata.jsonl`（含离线打分元数据）。
- **CN-GongWen-DataQA**：结构化记录 + Python 确定性答案 + 证据行 + 计算说明 + 异常标签 + 优先级排序 +
  接地播报 + 政策领域/医疗子领域分类（16 类任务，含 7 类进阶难题、5 种播报子类型）。
- **CN-GongWen-Writing**：按**目标产出 token** 分短(≤300)/中(300–800)/长(1200–2500)三档，覆盖 15 文种，
  蕴含复杂行文框架与规则（标题三要素、层次序数、上行文签发人、请示一文一事单一主送、报告不夹带请示、函为平行文）、
  **内容可执行性**（依据—目标—任务—责任—时限—保障—反馈）、**标点/数字规范**（GB/T 15834/15835）与**语言安全**。
  评分 rubric 与参考公文为**确定性事实接地**（金标准自评满分）。**医疗题**叠加医学合规规则与 2024–2026 医疗法规依据。
- **CN-GongWen-Audit**：在确定性"正确底稿"上**确定性注入**违规子集（约 1/4 完全合规对照），覆盖
  **16 类违规**——通用 11 类 + **5 类医疗专属**（夸大疗效、患者隐私、科研当临床、AI 替代医师、医保基金违规）。
  金标准由**独立检测器**校验诚实性。两个子任务：**找错** `--dataset audit`、**纠错改写** `--dataset rewrite`。

完整数据集设计见 [`gongwen_benchmark/README.md`](gongwen_benchmark/README.md)。

## 安装

核心生成/校验/打分**仅需标准库**，开箱即用。模型评测与 LLM 改写按需安装可选依赖：

```bash
pip install '.[llm]'            # LiteLLM（OpenAI / Anthropic / Qwen / DeepSeek / Together …）
pip install '.[azure]'         # Azure OpenAI（经 LiteLLM 的 azure/ 前缀路由）
pip install '.[poe]'           # Poe API（fastapi-poe）
pip install '.[all-providers]' # 以上全部
pip install '.[eval]'          # 评测排行榜全家桶：litellm + fastapi-poe + PyYAML（--models 需要）
pip install '.[dev]'           # pytest + PyYAML（跑测试）
```

---

# 一、生成测试数据（公文试题生成）

代表性提交档（standard：37 机关、8 个工作日；Q 600 / DataQA 1000 / 语料 4725 / Writing 90 / Audit 90）：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard
```

| Profile | 机关 | 工作日 | 用途 |
|---|---:|---:|---|
| `mini` | 5 | 2 | 快速冒烟 |
| `standard` | 37 | 8 | 提交的代表性基准 |
| `full` | 37 | 30 | 生产规模本地生成 |

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile full --q-count 1000 --dataqa-questions 3000
```

## 用 LLM 改写并"冻结"（可选，覆盖三套数据集的表层文本）

`--use-llm` 仅在**事实护栏**下改写**表层文本**——CN-GongWen-Q 的**问题措辞**、CN-GongWen-DataQA 的
**播报语言**、CN-GongWen-Writing 的**测试 prompt**；文种、字号、日期、密级、数值、排序、合规判定，
以及写作 rubric / 参考公文、审核金标准，**始终由 Python 确定性计算**。LLM 调用带磁盘缓存与重试，
**单条失败自动回退确定性模板**（冻结流程不中断）；未配密钥时启用会**提前明确报错**。运行后 `git add`
提交即"**冻结**"这批 LLM 产物；无 `--use-llm` 的逐字节复现仍以确定性基线为准。

三种 provider 任选其一（`--use-litellm` 为兼容旧用法的别名）：

```bash
# LiteLLM（OpenAI / Anthropic / Qwen / DeepSeek / Minimax 中继 …）
export LLM_API_KEY=sk-...
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider litellm --llm-model openai/gpt-4o-mini

# Azure OpenAI
export AZURE_API_KEY=...  AZURE_API_BASE=https://<resource>.openai.azure.com/
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider azure --llm-model <你的-deployment-名>

# Poe
export POE_API_KEY=...
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard \
  --use-llm --provider poe --llm-model GPT-4o-Mini
```

写作 prompt 生成器同样支持多 provider：

```bash
python gongwen_benchmark/scripts/generate_writing_prompts.py --count 90 \
  --use-llm --provider poe --llm-model Claude-3-7-Sonnet
```

---

# 二、评测你的模型（公文生成模型测试评估）

`eval_runner.py` 是端到端的**多模型评测器**：把任意模型跑过四套数据集 → 保存逐题预测 → 调用打分器 →
汇总成**带 95% 置信区间的排行榜**。支持 **Poe / Azure / LiteLLM** 三类后端，**断点续跑**（逐题落盘，
中断后重跑自动跳过已答题），并可用 `--limit N` 做小样本冒烟/控成本。

```bash
pip install '.[eval]'
```

## 单模型评测

```bash
# LiteLLM：OpenAI / Anthropic / Qwen / DeepSeek …
export OPENAI_API_KEY=sk-...
python gongwen_benchmark/scripts/eval_runner.py \
  --provider litellm --model openai/gpt-4o \
  --dataset all --output-dir results/gpt4o

# Azure OpenAI
export AZURE_API_KEY=...  AZURE_API_BASE=https://<resource>.openai.azure.com/
python gongwen_benchmark/scripts/eval_runner.py \
  --provider azure --model <deployment 名> --api-version 2024-08-01-preview \
  --dataset all --output-dir results/azure-gpt4o

# Poe
export POE_API_KEY=...
python gongwen_benchmark/scripts/eval_runner.py \
  --provider poe --model Claude-3-7-Sonnet \
  --dataset q --output-dir results/claude-poe

# 小样本冒烟（每套只评前 5 题，便宜地验证链路）
python gongwen_benchmark/scripts/eval_runner.py \
  --provider litellm --model openai/gpt-4o-mini --limit 5 \
  --dataset all --output-dir results/smoke
```

`--dataset` 取值：`q` / `dataqa` / `writing` / `audit` / `rewrite` / `all`（默认 `all`）。
每个模型的预测与 `model_meta.json` 写入 `results/<模型名>/`。

## 多模型排行榜（E7 Leaderboard）

用 YAML 列出一批模型（含强/中/弱梯队与三类 provider 示例，见
[`models_example.yaml`](gongwen_benchmark/scripts/models_example.yaml)）：

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
# 顺序评测所有模型，跑完自动出排行榜 results/leaderboard.json
python gongwen_benchmark/scripts/eval_runner.py \
  --models gongwen_benchmark/scripts/models_example.yaml \
  --dataset all --output-dir results/

# 只从已保存的预测重新汇总排行榜（不再调用模型）
python gongwen_benchmark/scripts/eval_runner.py --leaderboard-only --output-dir results/
```

排行榜按 `macro_avg`（四套数据集全部叶子指标的非加权均值）排序，并对各模型分数做 1000 次
bootstrap 重采样给出 95% 置信区间。

## 各 provider 环境变量

| Provider | 必填 | 可选 / 别名 |
|---|---|---|
| **litellm** | `LLM_API_KEY`（或 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` …） | `LLM_MODEL`、`LLM_API_BASE`（自建中继时填） |
| **azure** | `AZURE_API_KEY`、`AZURE_API_BASE` | `AZURE_DEPLOYMENT`、`AZURE_API_VERSION`（默认 `2024-08-01-preview`） |
| **poe** | `POE_API_KEY`（[poe.com/api_key](https://poe.com/api_key)） | `POE_BOT_NAME` |

通用：`CN_GW_LLM_TEMPERATURE`(0.35)、`CN_GW_LLM_TIMEOUT`、`CN_GW_LLM_RETRIES`(3)、
`CN_GW_LLM_CACHE`(缓存目录)、`CN_GW_EVAL_DELAY`(调用间隔秒)、`CN_GW_BOOTSTRAP_N`(1000)、
`CN_GW_CONTEXT_ROW_CAP`(DataQA 上下文行上限，0=不限；小上下文模型可设上限)、
`CN_GW_EVAL_PREFLIGHT`(默认 1；设 0 跳过启动前的一次探测调用)。

> **可靠性**：评测启动前会做 **provider 预检**（库/密钥/端点 + 一次探测调用），
> 失败即非 0 退出并跳过该模型——损坏环境不会以静默 ~0.57 混入排行榜。逐题预测**即时落盘**，
> 加 `--resume` 可断点续跑（默认覆盖重跑）。

## 底层打分器（eval_runner 已自动调用；也可单独使用）

五个打分器以金标准自评作连通性检查，理想分数为 1.0：

```bash
# 1) CN-GongWen-Q       预测: {"question_id","target_doc_type","expected_query_type","requires_clarification","should_refuse"}
python gongwen_benchmark/evaluation/scorer.py --dataset q \
  --gold gongwen_benchmark/dataset_1_question_only/questions_with_hidden_metadata.jsonl --pred your_q.jsonl
# 2) CN-GongWen-DataQA  预测: {"question_id","answer_value","evidence_rows"}
python gongwen_benchmark/evaluation/scorer.py --dataset dataqa \
  --gold gongwen_benchmark/dataset_2_data_qa/answers.jsonl --pred your_dataqa.jsonl
# 3) CN-GongWen-Writing 预测: {"question_id","answer":"<公文全文>"}
python gongwen_benchmark/evaluation/scorer.py --dataset writing \
  --gold gongwen_benchmark/dataset_3_writing/writing_prompts_with_rubric.jsonl --pred your_writing.jsonl
# 4) CN-GongWen-Audit·找错  预测: {"question_id","violations":["code",...]}
python gongwen_benchmark/evaluation/scorer.py --dataset audit \
  --gold gongwen_benchmark/dataset_4_audit/audit_tasks_with_gold.jsonl --pred your_audit.jsonl
# 5) CN-GongWen-Audit·改写  预测: {"question_id","rewrite":"<改写后的公文全文>"}
python gongwen_benchmark/evaluation/scorer.py --dataset rewrite \
  --gold gongwen_benchmark/dataset_4_audit/audit_tasks_with_gold.jsonl --pred your_rewrite.jsonl
```

---

## 校验与单元测试

```bash
python gongwen_benchmark/scripts/validate_artifacts.py   # 严格跨文件校验（四套数据集 + 金标准诚实性）
pytest -q
```

严格校验器检查 public/hidden 隔离、跨文件 ID、证据完整性、答案契约、任务/题型覆盖、难度分布
（hard ≥40%、easy ≤25%）、陷阱多样性（≥12）、医疗占比（≈50%）、写作参考公文 token 落桶、
审核金标准诚实性（注入即可检出、改写底稿本身零违规）等。

## 经批准的真实/混合数据

对经批准的脱敏聚合台账，使用 `--records-input` 传入；导入器拒绝个人隐私字段并默认匿名化机关。
完整工作流见 [`gongwen_benchmark/README.md`](gongwen_benchmark/README.md)。

## 目录结构

```
gongwen_benchmark/
├─ dataset_1_question_only/    # CN-GongWen-Q：public + hidden + taxonomy
├─ dataset_2_data_qa/          # CN-GongWen-DataQA：records.csv + questions/answers/...
├─ dataset_3_writing/          # CN-GongWen-Writing：public prompt + with_rubric(参考公文) + taxonomy
├─ dataset_4_audit/            # CN-GongWen-Audit：public(待审/待改写) + with_gold(违规+合规底稿) + taxonomy
├─ evaluation/                 # scorer.py（5 个打分器）+ 度量/规则/报告模板
├─ workflow/                   # 公文办理流转（OA）事件流示例
└─ scripts/
   ├─ generate_benchmarks.py   # 主生成器（Q + DataQA + 语料）
   ├─ generate_writing_prompts.py  # 写作测试 prompt 生成器
   ├─ generate_audit_tasks.py  # 审核/纠错任务生成器
   ├─ eval_runner.py           # ★ 多模型评测器 + 排行榜（Poe/Azure/LiteLLM）
   ├─ llm_providers.py         # ★ 统一 LLM provider 抽象（缓存/重试/事实护栏）
   ├─ models_example.yaml      # ★ 多模型评测配置示例
   ├─ litellm_minimax.py       # 向后兼容 shim（再导出 llm_providers）
   ├─ benchmark_schema.py      # 文种/要素/医疗三级分类/陷阱等 schema
   ├─ data_sources.py          # 经批准真实数据导入（拒隐私字段、匿名化）
   └─ validate_artifacts.py    # 严格跨文件校验器
```

## 二进制工件策略

PR 评审不支持二进制 diff，故仓库仅提交可审阅的文本工件（CSV / JSONL）。需要 Parquet 时本地生成：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py --profile standard --export-parquet /tmp/gongwen-records.parquet
```
