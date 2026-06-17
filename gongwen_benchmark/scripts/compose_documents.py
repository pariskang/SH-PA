"""Prompt-driven official-document composition via MiniMax (two-stage).

实际测试流程：明确的 prompt → 第一步生成「写作框架」(outline) → 第二步据框架生成「整体公文」。
覆盖不同文种（15 种法定公文）与不同长度（短/中/长）、不同政策方向（通用政务 / 医疗卫生及其具体主题）。

- 有 MINIMAX_API_KEY 时：真正调用 MiniMax（OpenAI 兼容）分两步写作。
- 无 Key 时：回退到确定性模板，保证离线可运行并可提交可审阅样例。

凭证读取环境变量（与 Colab 凭证单元一致）：
    MINIMAX_API_KEY, MINIMAX_API_BASE(默认 https://api.minimaxi.com/v1), MINIMAX_MODEL(默认 MiniMax-M1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_schema import DOC_TYPES, MEDICAL_AREAS, doc_type_by_name
from generate_benchmarks import ACTION_VERBS, DIR_CN, SUBJECTS, agency_metadata, hashed, pick
from litellm_minimax import LiteLLMConfig

ROOT = SCRIPT_DIR.parent
CACHE_DIR = Path(os.getenv("CN_GW_COMPOSE_CACHE", ".cache/gongwen_compose"))

# 不同长度规格：字数区间与正文层次数量
LENGTHS: dict[str, dict[str, Any]] = {
    "short": {"label": "短", "words": (300, 500), "sections": 2},
    "medium": {"label": "中", "words": (600, 900), "sections": 3},
    "long": {"label": "长", "words": (1200, 1800), "sections": 5},
}

# 文种结尾用语
CLOSING = {
    "通知": "特此通知。", "通报": "特此通报。", "公告": "特此公告。", "通告": "特此通告。",
    "请示": "以上请示妥否，请批复。", "报告": "特此报告。", "批复": "此复。",
    "意见": "请结合实际，认真贯彻落实。", "函": "特此函达，请予支持并函复。", "决定": "特此决定。",
    "决议": "特此决议。", "命令": "本命令自发布之日起施行。", "公报": "特此公报。",
    "议案": "以上议案，请予审议。", "纪要": "请各有关单位认真抓好落实。",
}

# 按文种类别提供正文小标题候选与通用要点（确定性回退用）
SECTION_TITLES = {
    "指示性": ("总体要求", "主要任务", "保障措施", "组织实施", "工作要求"),
    "呈请性": ("基本情况", "主要做法", "存在问题", "下一步打算", "请示事项"),
    "商洽性": ("商洽缘由", "具体事项", "工作建议", "请予支持"),
    "公布性": ("事项内容", "适用范围", "执行要求"),
    "记录性": ("会议概况", "议定事项", "责任分工", "工作要求"),
}
POINT_BANK = (
    "提高政治站位，统一思想认识，切实增强工作的责任感和紧迫感",
    "明确目标任务，细化工作举措，逐项明确责任单位和完成时限",
    "健全工作机制，加强统筹协调，形成上下联动、齐抓共管的工作格局",
    "强化要素保障，落实经费、人员和技术支撑，确保各项任务落到实处",
    "加强督促检查，建立台账管理和定期调度机制，及时发现和解决问题",
    "注重宣传引导，及时总结推广好经验好做法，营造良好工作氛围",
    "严格规范程序，依法依规推进，坚决守住安全和廉政底线",
)


@dataclass(frozen=True)
class DocSpec:
    spec_id: str
    doc_type: str          # 文种
    category: str          # 公文类别
    policy_domain: str     # 通用政务 / 医疗卫生
    medical_area: str
    medical_topic: str
    subject: str           # 事由
    agency: str            # 发文机关
    recipient: str         # 主送机关
    direction: str         # upward / downward / parallel
    length: str            # short / medium / long
    security: str
    urgency: str

    def prompt_brief(self) -> str:
        spec = LENGTHS[self.length]
        lo, hi = spec["words"]
        dom = self.policy_domain + (f"／{self.medical_area}／{self.medical_topic}" if self.medical_topic else "")
        return (f"文种：{self.doc_type}；发文机关：{self.agency}；行文方向：{DIR_CN[self.direction]}；"
                f"主送机关：{self.recipient or '（公布性公文可不标主送）'}；事由：{self.subject}；"
                f"政策方向：{dom}；目标长度：{spec['label']}（约{lo}-{hi}字，正文{spec['sections']}个层次）；"
                f"紧急程度：{self.urgency}；密级：{self.security}")


def _recipient(direction: str, doc_type: str, agency: str, level: str) -> str:
    if doc_type in ("公告", "通告", "公报", "命令", "决议"):
        return ""
    if direction == "upward":
        return "示范市人民代表大会常务委员会" if doc_type == "议案" else ("国务院" if level == "省级" else "示范市人民政府")
    if direction == "parallel":
        return "示范市财政局"
    return "各县（市、区）人民政府，各有关单位"


# 医疗子领域 → 对口发文机关代字（提升真实性；未列出的默认卫健委/疾控）
AREA_TO_AGENCY_CODE = {
    "医保管理": "示市医保", "医疗服务价格": "示市医保",
    "医药供应与集采": "示市药监", "药品器械监管": "示市药监",
    "中医药": "示市中医药", "职业健康": "示市卫监", "医疗质量与安全": "示市卫监",
    "公共卫生": "示市疾控",
}


def _agency_for(doc_type: str, medical: bool, area_name: str, agencies: list[dict[str, Any]], by_code: dict[str, dict], i: int):
    """文种-机关合法性：命令/议案仅政府；决议/公报由综合机关。其余医疗公文优先对口卫生机关，
    约 1/4 由综合机关（政府/办公厅）发文以体现三医联动。"""
    comp = [a for a in agencies if a["agency_category"] in ("政府", "政府办公厅(室)", "党委")]
    gov = [a for a in agencies if a["agency_category"] == "政府"]
    if doc_type in ("命令", "议案"):
        return gov[hashed("g", i) % len(gov)]
    if doc_type in ("决议", "公报"):
        return comp[hashed("c2", i) % len(comp)]
    if not medical:
        return agencies[hashed("ag", i) % len(agencies)]
    if hashed("compre", i) % 4 == 0:
        return comp[hashed("c", i) % len(comp)]
    code = AREA_TO_AGENCY_CODE.get(area_name, "示市卫健")
    return by_code.get(code) or by_code.get("示市卫健") or agencies[0]


def build_specs(count: int) -> list[DocSpec]:
    agencies = agency_metadata(37)
    by_code = {a["agency_code"]: a for a in agencies}
    specs: list[DocSpec] = []
    for i in range(count):
        dt = DOC_TYPES[i % len(DOC_TYPES)]               # 轮转覆盖全部 15 文种
        length = ("short", "medium", "long")[i % 3]      # 轮转覆盖三种长度
        medical = (i % 2 == 0)                           # 约一半医疗
        direction = dt.direction if dt.direction != "flexible" else "downward"
        if medical:
            area = MEDICAL_AREAS[hashed("ar", i) % len(MEDICAL_AREAS)]
            topic = area.topics[hashed("tp", i) % len(area.topics)]
            subject = pick(ACTION_VERBS, "v", i) + topic
            pd, ma, mt = "医疗卫生", area.name, topic
        else:
            area = None
            subject = SUBJECTS[hashed("sj", i) % len(SUBJECTS)]
            pd, ma, mt = "通用政务", "", ""
        ag = _agency_for(dt.name, medical, ma, agencies, by_code, i)
        security = "公开" if dt.category == "公布性" else ("秘密" if hashed("sec", i) % 5 == 0 else "公开")
        urgency = ("平件", "加急", "特急")[hashed("urg", i) % 6 % 3]
        specs.append(DocSpec(
            spec_id=f"DOC{i + 1:04d}", doc_type=dt.name, category=dt.category,
            policy_domain=pd, medical_area=ma, medical_topic=mt, subject=subject,
            agency=ag["agency_name"], recipient=_recipient(direction, dt.name, ag["agency_name"], ag["agency_level"]),
            direction=direction, length=length, security=security, urgency=urgency,
        ))
    return specs


# --------------------------------------------------------------------------------------
# 明确的 prompt（两阶段）
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = "你是资深党政机关公文写作专家，严格遵循《党政机关公文处理工作条例》(2012)与GB/T 9704—2012《党政机关公文格式》。"
SAFE_RULES = ("不得编造具体统计数字、法规条款号、真实机关或个人姓名；确需示例数据时以“〔示例〕”标注；"
              "标题须由发文机关名称、事由和文种组成；正文结构层次序数依次为“一、”“（一）”“1.”。")


def framework_messages(spec: DocSpec) -> list[dict[str, str]]:
    info = LENGTHS[spec.length]
    lo, hi = info["words"]
    user = (
        "第一步：请只生成【写作框架】（提纲，不要写正文全文），以JSON对象输出。\n"
        f"写作要求：{spec.prompt_brief()}\n"
        f"请输出JSON：{{\"标题\":\"...\",\"主送机关\":\"...\",\"正文结构\":[{{\"小标题\":\"...\",\"要点\":[\"...\"]}}],"
        f"\"落款\":{{\"发文机关署名\":\"...\",\"成文日期\":\"...\"}},\"预计字数\":{(lo + hi) // 2}}}\n"
        f"正文结构需约{info['sections']}个层次，每层2-4个要点。{SAFE_RULES}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def document_messages(spec: DocSpec, framework: dict[str, Any]) -> list[dict[str, str]]:
    info = LENGTHS[spec.length]
    lo, hi = info["words"]
    user = (
        f"第二步：请依据下面的【写作框架】撰写完整的《{spec.doc_type}》全文，遵循GB/T 9704—2012格式。\n"
        f"写作框架：{json.dumps(framework, ensure_ascii=False)}\n"
        f"要求：完整呈现标题、主送机关（如适用）、正文（层次序数“一、（一）1.”）、发文机关署名、成文日期；"
        f"目标字数约{lo}-{hi}字；文风庄重规范，契合{spec.doc_type}的语体与{DIR_CN[spec.direction]}行文方向；"
        f"{SAFE_RULES}仅输出公文文本本身，不要任何解释或额外说明。"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------------------
# LLM 调用（OpenAI 兼容；优先 openai SDK，回退 litellm；带磁盘缓存与指数退避）
# --------------------------------------------------------------------------------------
def _provider_chat(messages: list[dict[str, str]], cfg: LiteLLMConfig, json_mode: bool) -> str:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        import openai  # OpenAI 兼容客户端（推荐）
        client = openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        resp = client.chat.completions.create(
            model=cfg.model, messages=messages, temperature=cfg.temperature, timeout=cfg.timeout, **extra)
        return resp.choices[0].message.content
    except ImportError:
        import litellm
        resp = litellm.completion(
            model=cfg.litellm_model, api_base=cfg.api_base, api_key=cfg.api_key,
            messages=messages, temperature=cfg.temperature, timeout=cfg.timeout, **extra)
        return resp["choices"][0]["message"]["content"]


def llm_chat(messages: list[dict[str, str]], cfg: LiteLLMConfig, json_mode: bool = False) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(
        {"m": cfg.model, "b": cfg.api_base, "j": json_mode, "msg": messages}, ensure_ascii=False, sort_keys=True
    ).encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    last: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            out = _provider_chat(messages, cfg, json_mode)
            if json_mode:
                json.loads(out)  # 校验 JSON
            cache.write_text(out, encoding="utf-8")
            return out
        except Exception as exc:  # 网络/服务/JSON 失败一并重试
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"MiniMax 调用在 {cfg.retries} 次后失败：{last}")


# --------------------------------------------------------------------------------------
# 确定性回退（无 API Key 时离线生成框架与全文）
# --------------------------------------------------------------------------------------
def _example_date(spec: DocSpec) -> str:
    day = hashed("date", spec.spec_id) % 28 + 1
    return f"2026年6月{day}日"


def deterministic_framework(spec: DocSpec) -> dict[str, Any]:
    info = LENGTHS[spec.length]
    titles = SECTION_TITLES.get(spec.category, SECTION_TITLES["指示性"])
    n = min(info["sections"], len(titles))
    structure = []
    for s in range(n):
        pts = [POINT_BANK[(hashed("pt", spec.spec_id, s, j) % len(POINT_BANK))] for j in range(2 if spec.length == "short" else 3)]
        structure.append({"小标题": titles[s], "要点": list(dict.fromkeys(pts))})
    return {
        "标题": f"{spec.agency}关于{spec.subject}的{spec.doc_type}",
        "主送机关": spec.recipient,
        "正文结构": structure,
        "落款": {"发文机关署名": spec.agency, "成文日期": _example_date(spec)},
        "预计字数": sum(info["words"]) // 2,
    }


CN_NUM = "一二三四五六七八九十"


def _lead(spec: DocSpec) -> str:
    if spec.category == "呈请性":
        return (f"现将{spec.subject}有关情况报告如下："
                if spec.doc_type == "报告" else f"为{spec.subject}，现请示如下：")
    if spec.category == "商洽性":
        return f"为{spec.subject}，现就有关事项商洽如下："
    if spec.category == "公布性":
        return f"现就{spec.subject}有关事项{spec.doc_type}如下："
    if spec.category == "记录性":
        return f"会议研究了{spec.subject}有关工作，现将议定事项纪要如下："
    verb = {"通知": "通知", "意见": "意见", "决定": "决定", "通报": "通报", "批复": "批复"}.get(spec.doc_type, "通知")
    return f"为{spec.subject}，根据有关工作部署，结合本地实际，现就有关事项{verb}如下："


def deterministic_document(spec: DocSpec, framework: dict[str, Any]) -> str:
    lines = [framework["标题"], ""]
    if framework.get("主送机关"):
        lines += [f"{framework['主送机关']}：", ""]
    lines.append("　　" + _lead(spec))
    for idx, sec in enumerate(framework["正文结构"]):
        lines.append("")
        lines.append(f"　　{CN_NUM[idx]}、{sec['小标题']}")
        for j, pt in enumerate(sec["要点"]):
            prefix = f"（{CN_NUM[j]}）" if spec.length != "short" else ""
            lines.append(f"　　{prefix}{pt}。")
    lines += ["", "　　" + CLOSING.get(spec.doc_type, "特此通知。"), "",
              "　　　　　　　　　　" + framework["落款"]["发文机关署名"],
              "　　　　　　　　　　" + framework["落款"]["成文日期"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 两阶段编排
# --------------------------------------------------------------------------------------
def compose(spec: DocSpec, cfg: LiteLLMConfig | None, use_llm: bool) -> dict[str, Any]:
    if use_llm and cfg is not None:
        framework = json.loads(llm_chat(framework_messages(spec), cfg, json_mode=True))
        framework.setdefault("落款", {"发文机关署名": spec.agency, "成文日期": _example_date(spec)})
        document = llm_chat(document_messages(spec, framework), cfg, json_mode=False).strip()
        engine = "minimax"
    else:
        framework = deterministic_framework(spec)
        document = deterministic_document(spec, framework)
        engine = "deterministic"
    return {"spec": asdict(spec), "framework": framework, "document": document, "engine": engine}


def main() -> None:
    parser = argparse.ArgumentParser(description="基于明确 prompt 的两阶段公文生成（写作框架→整体公文）")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--out", type=Path, default=ROOT / "generated_samples")
    parser.add_argument("--no-llm", action="store_true", help="强制使用确定性回退（不调用 MiniMax）")
    parser.add_argument("--md-samples", type=int, default=6, help="另存为单独 .md 文件的篇数")
    args = parser.parse_args()

    has_key = bool(os.getenv("MINIMAX_API_KEY"))
    use_llm = has_key and not args.no_llm
    cfg = LiteLLMConfig() if use_llm else None

    specs = build_specs(args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        result = compose(spec, cfg, use_llm)
        records.append(result)
    with (args.out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records[: args.md_samples]:
        s = r["spec"]
        name = f"{s['spec_id']}_{s['doc_type']}_{LENGTHS[s['length']]['label']}.md"
        (args.out / name).write_text(
            f"<!-- {s['policy_domain']} {s['medical_topic']} | 文种={s['doc_type']} 长度={s['length']} "
            f"engine={r['engine']} -->\n\n# 写作框架\n\n```json\n"
            f"{json.dumps(r['framework'], ensure_ascii=False, indent=2)}\n```\n\n# 整体公文\n\n{r['document']}\n",
            encoding="utf-8")
    print(json.dumps({
        "count": len(records), "engine": records[0]["engine"] if records else "n/a",
        "use_llm": use_llm, "out": str(args.out),
        "doc_types": sorted({r["spec"]["doc_type"] for r in records}),
        "lengths": sorted({r["spec"]["length"] for r in records}),
        "medical_share": round(sum(r["spec"]["policy_domain"] == "医疗卫生" for r in records) / max(1, len(records)), 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
