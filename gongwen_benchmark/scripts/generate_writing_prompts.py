"""CN-GongWen-Writing：按目标产出 token 分桶的公文「写作测试 prompt」生成器（dataset_3）。

设计要点（对应仓库“零随机、逐字节复现”的灵魂与本次需求）：
- 每条测试 prompt 都绑定一个**确定性规格**（文种 / 行文方向 / 发文机关 / 主送机关 / 事由 /
  长度分桶 / 复杂行文框架 / 须规避雷区），规格集合由 SHA-256 派生，跨环境一致。
- **长度按目标产出 token 分桶**（short / medium / long），用确定性估算器 ``estimate_tokens``，
  与校验、打分共用同一口径。
- **测试 prompt 文本**：有 MINIMAX_API_KEY 时由 LLM **一次生成 10 条**（JSON 数组、逐条
  schema 校验、事实护栏、不合格回退、补足到 10）；无 key 时用确定性模板，保证离线/CI 可跑。
  提交入库即“冻结”，复现 = 复用已提交文件。
- **评分真相（rubric / framework / reference_answer）始终确定性生成**、事实接地，因此
  逐字节可复现，且金标准自评满分。LLM 永不决定文种、字号、日期、密级等关键事实。

遵循《党政机关公文处理工作条例》(2012) 与 GB/T 9704—2012《党政机关公文格式》。
所有机关、字号、人名均为合成示例，不对应任何真实单位或公文。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_schema import (
    DOC_TYPE_GUIDE,
    DOC_TYPES,
    EXECUTABLE_DOC_TYPES,
    FORBIDDEN_PHRASES,
    MEDICAL_AREAS,
    SECRET_LEVELS,
    SELF_CHECK,
    STANDARDS,
    TRAP_TYPES,
    doc_type_by_name,
)
from generate_benchmarks import ACTION_VERBS, DIR_CN, SUBJECTS, agency_metadata, hashed, pick, write_jsonl
from litellm_minimax import LiteLLMConfig, completion_json
from tokens import estimate_tokens

ROOT = SCRIPT_DIR.parent
DATASET3 = ROOT / "dataset_3_writing"

# --- 长度分桶：按“目标产出 token”划分（确定性估算口径）---
LENGTH_BUCKETS: dict[str, dict[str, Any]] = {
    "short":  {"label": "短", "target_tokens": (150, 300),  "min_sections": 2},
    "medium": {"label": "中", "target_tokens": (300, 800),  "min_sections": 3},
    "long":   {"label": "长", "target_tokens": (1200, 2500), "min_sections": 5},
}
LENGTH_ORDER = ("short", "medium", "long")

SYSTEM_PROMPT = (
    "你是资深党政机关公文写作与评测命题专家，严格遵循《党政机关公文处理工作条例》(2012)"
    "与 GB/T 9704—2012《党政机关公文格式》。"
)

# 文种结尾用语（确定性参考公文用）
CLOSING = {
    "通知": "特此通知。", "通报": "特此通报。", "公告": "特此公告。", "通告": "特此通告。",
    "请示": "以上请示妥否，请批复。", "报告": "特此报告。", "批复": "此复。",
    "意见": "请结合实际，认真贯彻落实。", "函": "特此函达，请予支持并函复。", "决定": "特此决定。",
    "决议": "特此决议。", "命令": "本命令自发布之日起施行。", "公报": "特此公报。",
    "议案": "以上议案，请予审议。", "纪要": "请各有关单位认真抓好落实。",
}

# 各文种类别的正文小标题候选（≥6/类，保证长文 5 层不重复）
SECTION_TITLES = {
    "指示性": ("总体要求", "主要任务", "重点举措", "保障措施", "组织实施", "工作要求", "督促落实"),
    "呈请性": ("基本情况", "主要做法", "存在问题", "原因分析", "下一步打算", "请示事项", "保障建议"),
    "商洽性": ("商洽缘由", "具体事项", "工作建议", "协作方式", "请予支持", "联系安排"),
    "公布性": ("事项内容", "适用范围", "执行要求", "施行时间", "解释说明", "其他事项"),
    "记录性": ("会议概况", "议定事项", "责任分工", "时限要求", "保障措施", "工作要求"),
}

# 正文要点库（确定性参考公文用；避免出现密级词与“第X条”法规引用，便于雷区规避校验）
POINT_BANK = (
    "提高政治站位，统一思想认识，切实增强工作的责任感和紧迫感",
    "明确目标任务，细化工作举措，逐项明确责任单位和完成时限",
    "健全工作机制，加强统筹协调，形成上下联动、齐抓共管的工作格局",
    "强化要素保障，落实经费、人员和技术支撑，确保各项任务落到实处",
    "加强督促检查，建立台账管理和定期调度机制，及时发现和解决问题",
    "注重宣传引导，及时总结推广好经验好做法，营造良好工作氛围",
    "严格规范程序，依法依规推进，坚决守住安全和廉政底线",
    "突出问题导向，聚焦堵点难点，分类施策、精准发力推动整改",
    "压实主体责任，强化考核激励，把任务完成情况纳入年度绩效管理",
    "坚持以人民为中心，畅通诉求反映渠道，切实提升群众满意度",
    "加强能力建设，组织业务培训和经验交流，提升队伍专业化水平",
    "推动信息化支撑，加强数据共享和业务协同，提高办理质效",
    "强化风险防控，完善应急预案，做到早发现、早报告、早处置",
    "注重示范引领，培育典型样板，以点带面推动整体提升",
    "加强部门协同，明确职责边界，建立信息互通和会商研判机制",
    "巩固工作成果，建立长效机制，防止反弹回潮、确保常态长效",
)

# 长文补足用的过渡句（确定性，按 spec+计数取用，控制篇幅落入目标 token 区间）
ELAB_BANK = (
    "各地区各部门要深刻认识本项工作的重要意义，切实把思想和行动统一到工作部署上来。",
    "要结合本地区本部门实际，制定切实可行的实施方案，确保各项要求落地见效。",
    "要加强组织领导，主要负责同志要亲自抓、负总责，分管负责同志要具体抓、抓到位。",
    "要细化分解任务，明确时间表和路线图，挂图作战、对账销号，确保按期完成。",
    "要强化协同配合，加强跨部门沟通会商，形成工作合力、避免推诿扯皮。",
    "要加强经费物资和人员力量保障，向重点环节和薄弱地区倾斜，夯实工作基础。",
    "要建立健全调度通报和督查问效机制，对工作推进不力的及时提醒、督促整改。",
    "要注重总结提炼可复制可推广的经验做法，加强宣传引导，营造良好社会氛围。",
    "要坚持稳中求进，处理好质量与进度、当前与长远的关系，确保工作行稳致远。",
    "要严守纪律规矩，规范工作流程，自觉接受监督，确保各项工作经得起检验。",
    "要密切跟踪掌握工作进展和苗头性倾向性问题，及时研究解决、动态优化措施。",
    "要充分发挥基层一线作用，问需于民、问计于民，不断提升工作的针对性和实效性。",
    "要加强统计分析和效果评估，用数据说话，以结果导向检验工作成色。",
    "要持续优化服务流程，减环节、压时限、提效率，切实方便企业和群众办事。",
    "要强化数字赋能，推动业务系统互联互通，以信息化手段提升治理效能。",
    "要健全容错纠错和正向激励机制，旗帜鲜明为担当者担当、为干事者撑腰。",
)

CN_NUM = "一二三四五六七八九十"


@dataclass(frozen=True)
class WritingSpec:
    spec_id: str
    doc_type: str            # 文种
    doc_type_code: str
    category: str            # 公文类别
    direction: str           # upward / downward / parallel（flexible 文种已落为具体方向）
    needs_signatory: bool     # 上行文须标注签发人
    single_recipient: bool    # 主送须单一（请示尤甚）
    policy_domain: str        # 通用政务 / 医疗卫生
    medical_area: str
    medical_topic: str
    subject: str             # 事由
    agency: str              # 发文机关
    agency_code: str         # 机关代字
    agency_level: str
    recipient: str           # 主送机关（公布性公文可为空）
    length: str              # short / medium / long
    security: str            # 公开 / 秘密 / 机密 / 绝密
    urgency: str             # 平件 / 加急 / 特急
    has_attachment: bool
    has_cc: bool


# --------------------------------------------------------------------------------------
# 确定性规格构建（SHA-256 派生，零随机）
# --------------------------------------------------------------------------------------
def _recipient(direction: str, doc_type: str, level: str) -> str:
    """按行文方向与文种给出合规主送机关；公布性公文可不标主送。"""
    if doc_type in ("公告", "通告", "公报", "命令", "决议"):
        return ""
    if direction == "upward":
        if doc_type == "议案":
            return "示范市人民代表大会常务委员会"
        return "国务院" if level == "省级" else "示范市人民政府"  # 上行文：单一上级
    if direction == "parallel":
        return "示范市财政局"  # 平行文：不相隶属机关
    return "各县（市、区）人民政府，各有关单位"  # 下行文：下级机关


_AREA_TO_CODE = {
    "医保管理": "示市医保", "医疗服务价格": "示市医保",
    "医药供应与集采": "示市药监", "药品器械监管": "示市药监",
    "中医药": "示市中医药", "职业健康": "示市卫监", "医疗质量与安全": "示市卫监",
    "公共卫生": "示市疾控",
}


def _agency_for(doc_type: str, medical: bool, area_name: str,
                agencies: list[dict[str, Any]], by_code: dict[str, dict], i: int) -> dict[str, Any]:
    """文种—机关合法性：命令/议案仅政府；决议/公报由综合机关；医疗公文优先对口卫生机关，
    约 1/4 由综合机关（政府/办公厅/党委）发文以体现“三医联动”。"""
    comp = [a for a in agencies if a["agency_category"] in ("政府", "政府办公厅(室)", "党委")]
    gov = [a for a in agencies if a["agency_category"] == "政府"]
    if doc_type in ("命令", "议案"):
        return gov[hashed("wg", i) % len(gov)]
    if doc_type in ("决议", "公报"):
        return comp[hashed("wc2", i) % len(comp)]
    if not medical:
        return agencies[hashed("wag", i) % len(agencies)]
    if hashed("wcompre", i) % 4 == 0:
        return comp[hashed("wc", i) % len(comp)]
    code = _AREA_TO_CODE.get(area_name, "示市卫健")
    return by_code.get(code) or by_code.get("示市卫健") or agencies[0]


def _combinations() -> list[tuple[Any, str, bool]]:
    """全部 (文种 × 长度 × 医疗/通用) 组合；count=90 时恰好各覆盖一次。"""
    combos: list[tuple[Any, str, bool]] = []
    for dt in DOC_TYPES:
        for length in LENGTH_ORDER:
            for medical in (True, False):
                combos.append((dt, length, medical))
    return combos


def build_writing_specs(count: int) -> list[WritingSpec]:
    agencies = agency_metadata(37)
    by_code = {a["agency_code"]: a for a in agencies}
    combos = _combinations()
    specs: list[WritingSpec] = []
    for i in range(count):
        dt, length, medical = combos[i % len(combos)]
        direction = dt.direction if dt.direction != "flexible" else "downward"
        if medical:
            area = MEDICAL_AREAS[hashed("war", i) % len(MEDICAL_AREAS)]
            topic = area.topics[hashed("wtp", i) % len(area.topics)]
            subject = pick(ACTION_VERBS, "wv", i) + topic
            pd, ma, mt = "医疗卫生", area.name, topic
        else:
            subject = SUBJECTS[hashed("wsj", i) % len(SUBJECTS)]
            pd, ma, mt = "通用政务", "", ""
        ag = _agency_for(dt.name, medical, ma, agencies, by_code, i)
        security = "公开" if dt.category == "公布性" else ("秘密" if hashed("wsec", i) % 6 == 0 else "公开")
        urgency = ("平件", "平件", "加急", "特急")[hashed("wurg", i) % 4]
        recipient = _recipient(direction, dt.name, ag["agency_level"])
        specs.append(WritingSpec(
            spec_id=f"WP_{i + 1:06d}", doc_type=dt.name, doc_type_code=dt.code, category=dt.category,
            direction=direction, needs_signatory=dt.needs_signatory, single_recipient=dt.single_recipient,
            policy_domain=pd, medical_area=ma, medical_topic=mt, subject=subject,
            agency=ag["agency_name"], agency_code=ag["agency_code"], agency_level=ag["agency_level"],
            recipient=recipient, length=length, security=security, urgency=urgency,
            has_attachment=(length in ("medium", "long") and hashed("watt", i) % 2 == 0),
            has_cc=(direction != "upward" and length in ("medium", "long") and hashed("wcc", i) % 2 == 0),
        ))
    return specs


# --------------------------------------------------------------------------------------
# 复杂行文框架 / 评分 rubric（确定性、事实接地）
# --------------------------------------------------------------------------------------
def required_elements(spec: WritingSpec) -> list[str]:
    """GB/T 9704—2012 该题须呈现的格式要素代码（供结构化评分）。"""
    codes = {"E04", "E05", "E07", "E09", "E11", "E12"}  # 机关标志/字号/标题/正文/署名/成文日期
    if spec.needs_signatory:
        codes.add("E06")               # 上行文标签发人
    if spec.recipient:
        codes.add("E08")               # 主送机关
    if spec.security in SECRET_LEVELS:
        codes.update({"E01", "E02"})   # 涉密：份号 + 密级
    if spec.urgency != "平件":
        codes.add("E03")               # 紧急程度
    if spec.has_attachment:
        codes.update({"E10", "E15"})   # 附件说明 + 附件
    if spec.doc_type == "请示":
        codes.add("E14")               # 请示附注注明联系人电话
    if spec.doc_type != "纪要":
        codes.add("E13")               # 印章（纪要不盖章）
    if spec.has_cc:
        codes.add("E16")               # 抄送机关
    return sorted(codes)


def _rule_constraints(spec: WritingSpec) -> list[str]:
    """该文种须遵循的行文规则（“复杂逻辑”的硬约束）。"""
    rules = ["标题须由发文机关名称、事由和文种三要素组成，不得省略文种",
             "正文结构层次序数依次为“一、”“（一）”“1.”“（1）”，不得颠倒混用",
             "成文日期用阿拉伯数字标全年月日"]
    if spec.direction == "upward":
        rules.append("上行文须标注签发人，且只主送一个上级机关、不抄送下级机关")
    if spec.doc_type == "请示":
        rules += ["请示须一文一事、不得多头主送", "请示应在附注处注明联系人姓名和电话"]
    if spec.doc_type == "报告":
        rules.append("报告不得夹带请示事项")
    if spec.doc_type == "函":
        rules.append("函用于不相隶属机关之间，语气商洽、不得下达指令")
    if spec.doc_type == "意见" and spec.direction == "downward":
        rules.append("下行的意见为指导性文件，不得作为强制命令")
    if spec.doc_type in ("公告", "通告"):
        rules.append("公布性公文可不标注主送机关")
    if spec.security in SECRET_LEVELS:
        rules.append("涉密公文须标注份号与密级、保密期限")
    guide = DOC_TYPE_GUIDE.get(spec.doc_type)
    if guide:
        rules += list(guide.key_points)
    if spec.doc_type in EXECUTABLE_DOC_TYPES and spec.direction == "downward":
        rules.append("须写明责任单位、完成时限与报送/反馈要求，避免只有抽象表态（依据—目标—任务—责任—时限—保障—反馈）")
    rules += [
        "发文字号年份用六角括号〔〕，“（一）”“（1）”等层次序号后不加顿号（GB/T 15834）",
        "数字用法规范，发文顺序号不编虚位、不加“第”（GB/T 15835）",
        "语言稳准短实，不使用夸大、网络化或绝对化表述",
    ]
    return rules


def _forbidden_traps(spec: WritingSpec) -> list[str]:
    """该题须规避的雷区（取自 18 类陷阱，作为负向约束）。"""
    traps = ["fabricated_doc_number", "fabricated_legal_basis", "fabricated_security_level"]
    if spec.doc_type == "请示":
        traps.append("multi_head_qingshi")
    if spec.doc_type == "报告":
        traps.append("confuse_baogao_qingshi")
    if spec.doc_type == "意见":
        traps.append("treat_yijian_as_order")
    if spec.doc_type == "函":
        traps.append("fabricated_doc_type")
    if spec.policy_domain == "医疗卫生":
        traps.append("leak_real_agency_identity")
    return sorted(set(traps), key=traps.index)


def _pitfalls(spec: WritingSpec) -> list[str]:
    """该文种的典型雷区（自然语言，来自文种辨析表与政策政治表述雷区）。"""
    guide = DOC_TYPE_GUIDE.get(spec.doc_type)
    items = list(guide.pitfalls) if guide else []
    if spec.direction == "upward":
        items.append("主送上级领导个人或多头主送")
    items.append("把未出台事项写成已决定、把部门意见写成党委政府决定")
    return list(dict.fromkeys(items))


def build_framework(spec: WritingSpec) -> dict[str, Any]:
    """复杂写作框架（提纲）：标题 + 主送 + 多层次正文要点 + 落款。"""
    info = LENGTH_BUCKETS[spec.length]
    titles = SECTION_TITLES.get(spec.category, SECTION_TITLES["指示性"])
    n = min(info["min_sections"], len(titles))
    structure = []
    for s in range(n):
        k = 2 if spec.length == "short" else 3
        pts = [POINT_BANK[hashed("wpt", spec.spec_id, s, j) % len(POINT_BANK)] for j in range(k)]
        structure.append({"小标题": titles[s], "要点": list(dict.fromkeys(pts))})
    return {
        "标题": f"{spec.agency}关于{spec.subject}的{spec.doc_type}",
        "主送机关": spec.recipient,
        "正文结构": structure,
        "落款": {"发文机关署名": spec.agency, "成文日期": _example_date(spec)},
        "目标token区间": list(info["target_tokens"]),
    }


def build_rubric(spec: WritingSpec) -> dict[str, Any]:
    info = LENGTH_BUCKETS[spec.length]
    return {
        "target_doc_type": spec.doc_type_code,
        "doc_type_name": spec.doc_type,
        "direction": spec.direction,
        "length_bucket": spec.length,
        "target_tokens": list(info["target_tokens"]),
        "min_sections": info["min_sections"],
        "required_elements": required_elements(spec),
        "needs_signatory": spec.needs_signatory,
        "single_recipient": spec.single_recipient,
        "rule_constraints": _rule_constraints(spec),
        "forbidden_traps": _forbidden_traps(spec),
        "pitfalls": _pitfalls(spec),
        "require_executability": spec.doc_type in EXECUTABLE_DOC_TYPES and spec.direction == "downward",
        "forbidden_phrases": list(FORBIDDEN_PHRASES),
        "standards": list(STANDARDS),
    }


# --------------------------------------------------------------------------------------
# 确定性参考公文（合规、命中目标 token 区间，用作金标准与连通性自评）
# --------------------------------------------------------------------------------------
def _example_date(spec: WritingSpec) -> str:
    return f"2026年6月{hashed('wdate', spec.spec_id) % 28 + 1}日"


def _doc_number(spec: WritingSpec) -> str:
    return f"{spec.agency_code}〔2026〕{hashed('wseq', spec.spec_id) % 90 + 1}号"


def _lead(spec: WritingSpec) -> str:
    if spec.category == "呈请性":
        return (f"现将{spec.subject}有关情况报告如下："
                if spec.doc_type == "报告" else f"为{spec.subject}，现请示如下：")
    if spec.category == "商洽性":
        return f"为{spec.subject}，现就有关事项商洽如下："
    if spec.category == "公布性":
        return f"现就{spec.subject}有关事项{spec.doc_type}如下："
    if spec.category == "记录性":
        return f"会议研究了{spec.subject}有关工作，现将议定事项纪要如下："
    return f"为{spec.subject}，根据有关工作部署，结合本地实际，现就有关事项通知如下："


def deterministic_reference(spec: WritingSpec, framework: dict[str, Any]) -> str:
    """生成一篇合规公文：满足标题三要素、层次序数、署名/日期、上行文签发人、请示单一主送/附注，
    并通过补足过渡句使估算 token 落入目标区间。"""
    lo, hi = LENGTH_BUCKETS[spec.length]["target_tokens"]
    header: list[str] = []
    if spec.security in SECRET_LEVELS:
        header += ["000001", f"{spec.security}★长期"]   # 份号(E01) + 密级和保密期限(E02)
    if spec.urgency != "平件":
        header.append(spec.urgency)                       # 紧急程度(E03)
    header += [framework["标题"], _doc_number(spec)]
    if spec.needs_signatory:
        header.append("签发人：示例")
    header.append("")
    if framework.get("主送机关"):
        header += [f"{framework['主送机关']}：", ""]
    header.append("　　" + _lead(spec))

    tail: list[str] = []
    if spec.doc_type in EXECUTABLE_DOC_TYPES and spec.direction == "downward":
        d = hashed("wexec", spec.spec_id) % 27 + 1   # 可执行性：责任主体+完成时限+报送反馈
        tail.append(f"　　各责任单位要明确分工、压实责任，于2026年7月{d}日前完成，并将落实情况及时报送我机关。")
    if spec.has_attachment:
        tail.append("　　附件：1.（示例附件名称）")
    tail += ["", "　　" + CLOSING.get(spec.doc_type, "特此通知。")]
    if spec.doc_type == "请示":
        tail.append("　　（联系人：示例，联系电话：0000-00000000）")
    tail += ["", "　　　　　　　　　　" + framework["落款"]["发文机关署名"],
             "　　　　　　　　　　" + framework["落款"]["成文日期"]]
    if spec.has_cc:
        tail += ["", "抄送：示范市有关部门。"]

    # 正文：以 framework 要点为种子，再按目标 token 区间确定性增删，保证落入 [lo, hi]。
    n = len(framework["正文结构"])
    sec_titles = [sec["小标题"] for sec in framework["正文结构"]]
    sec_points = [list(sec["要点"]) for sec in framework["正文结构"]]
    point_pool = list(POINT_BANK) + list(ELAB_BANK)

    def render() -> str:
        body: list[str] = []
        for i in range(n):
            body.append(f"　　{CN_NUM[i]}、{sec_titles[i]}")
            for j, pt in enumerate(sec_points[i]):
                prefix = f"（{CN_NUM[j % 10]}）" if spec.length != "short" else ""
                body.append(f"　　{prefix}{pt}。")
        return "\n".join(header + body + tail)

    # 超上限：自后向前回收多余要点（每节至少留 1 个，保住 n 个一级层次）
    while estimate_tokens(render()) > hi:
        for i in range(n - 1, -1, -1):
            if len(sec_points[i]) > 1:
                sec_points[i].pop()
                break
        else:
            break
    # 不足下限：轮流补足确定性要点（受 hi 约束）
    counter = 0
    while estimate_tokens(render()) < lo and counter < 600:
        i = counter % n
        sec_points[i].append(point_pool[hashed("wp2", spec.spec_id, counter) % len(point_pool)])
        if estimate_tokens(render()) > hi:
            sec_points[i].pop()
            break
        counter += 1
    return render()


# --------------------------------------------------------------------------------------
# 测试 prompt 文本：确定性模板 / LLM 一次 10 条
# --------------------------------------------------------------------------------------
def deterministic_prompt(spec: WritingSpec) -> str:
    info = LENGTH_BUCKETS[spec.length]
    lo, hi = info["target_tokens"]
    dom = spec.policy_domain + (f"（{spec.medical_area}／{spec.medical_topic}）" if spec.medical_topic else "")
    guide = DOC_TYPE_GUIDE.get(spec.doc_type)
    usage = f"（{guide.usage}）" if guide else ""
    rules = "；".join(_rule_constraints(spec))
    traps = "、".join(_forbidden_traps(spec))
    pitfalls = "；".join(_pitfalls(spec))
    return (
        f"请撰写一篇《{spec.doc_type}》{usage}。发文机关：{spec.agency}；行文方向：{DIR_CN[spec.direction]}；"
        f"主送机关：{spec.recipient or '（公布性公文可不标主送）'}；事由：{spec.subject}；政策方向：{dom}。\n"
        f"目标篇幅：{info['label']}文，正文不少于{info['min_sections']}个层次，全文约 {lo}-{hi} tokens。\n"
        f"格式与行文要求：严格遵循{'、'.join(STANDARDS)}；{rules}。\n"
        f"须规避的雷区：{traps}；并注意：{pitfalls}（不得编造具体发文字号、法规条款、真实机关或个人姓名）。\n"
        f"仅输出公文文本本身（标题、主送机关、正文、附件说明如有、发文机关署名、成文日期）。"
    )


def _chunks(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _spec_brief(spec: WritingSpec) -> dict[str, Any]:
    info = LENGTH_BUCKETS[spec.length]
    return {
        "spec_id": spec.spec_id, "文种": spec.doc_type, "行文方向": DIR_CN[spec.direction],
        "发文机关": spec.agency, "主送机关": spec.recipient or "（公布性公文可不标主送）",
        "事由": spec.subject,
        "政策方向": spec.policy_domain + (f"（{spec.medical_topic}）" if spec.medical_topic else ""),
        "目标token区间": list(info["target_tokens"]), "正文最少层次": info["min_sections"],
        "行文规则": _rule_constraints(spec), "须规避雷区": _forbidden_traps(spec),
        "典型雷区": _pitfalls(spec), "须遵循标准": list(STANDARDS),
        "须含可执行要素": spec.doc_type in EXECUTABLE_DOC_TYPES and spec.direction == "downward",
    }


def _author_messages(batch: list[WritingSpec]) -> list[dict[str, str]]:
    items = [_spec_brief(s) for s in batch]
    user = (
        f"请为下列 {len(items)} 条写作任务各撰写【一条面向被测大模型的中文写作测试 prompt（指令）】。\n"
        "每条要求：① 自然连贯的中文指令；② 完整、忠实地包含该任务给定的文种、行文方向、发文机关、"
        "主送机关、事由、目标篇幅(token 区间)与正文层次数、行文规则与须规避雷区；③ 明确要求遵循"
        "《党政机关公文处理工作条例》与 GB/T 9704—2012；④ 不要替模型写出正文，"
        "不要新增或编造具体发文字号、法规条款号、真实机关或个人姓名。\n"
        f"输入任务（务必逐条对应 spec_id）：{json.dumps(items, ensure_ascii=False)}\n"
        '仅输出 JSON 对象：{"prompts":[{"spec_id":"...","prompt":"..."}, ...]}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _prompt_ok(spec: WritingSpec, candidate: str) -> bool:
    """事实护栏：LLM 改写后的指令仍须忠实包含关键事实，否则回退确定性模板。"""
    lo, _ = LENGTH_BUCKETS[spec.length]["target_tokens"]
    if not candidate or len(candidate) < 30:
        return False
    return all(tok in candidate for tok in (spec.doc_type, spec.agency, spec.subject, str(lo)))


def author_prompts(specs: list[WritingSpec], cfg: LiteLLMConfig | None, use_llm: bool) -> dict[str, tuple[str, str]]:
    """返回 spec_id -> (prompt 文本, engine)。LLM 一次 10 条；不合格逐条回退确定性模板。"""
    if not (use_llm and cfg is not None):
        return {s.spec_id: (deterministic_prompt(s), "deterministic") for s in specs}
    out: dict[str, tuple[str, str]] = {}
    for batch in _chunks(specs, 10):
        got: dict[str, str] = {}
        try:
            data = completion_json(_author_messages(batch), cfg)
            for d in data.get("prompts", []) if isinstance(data, dict) else []:
                if isinstance(d, dict) and d.get("spec_id"):
                    got[str(d["spec_id"])] = str(d.get("prompt", "")).strip()
        except Exception:
            got = {}
        for s in batch:
            cand = got.get(s.spec_id, "")
            out[s.spec_id] = (cand, "minimax") if _prompt_ok(s, cand) else (deterministic_prompt(s), "deterministic")
    return out


# --------------------------------------------------------------------------------------
# 数据集构建与落盘
# --------------------------------------------------------------------------------------
def build_writing_dataset(count: int, use_llm: bool = False, cfg: LiteLLMConfig | None = None):
    specs = build_writing_specs(count)
    prompts = author_prompts(specs, cfg, use_llm)
    public, hidden = [], []
    for s in specs:
        framework = build_framework(s)
        rubric = build_rubric(s)
        reference = deterministic_reference(s, framework)
        prompt, engine = prompts[s.spec_id]
        public.append({"question_id": s.spec_id, "prompt": prompt})
        hidden.append({
            "question_id": s.spec_id, "prompt": prompt, "length_bucket": s.length,
            "spec": asdict(s), "rubric": rubric, "framework": framework,
            "reference_answer": reference, "reference_tokens": estimate_tokens(reference),
            "prompt_engine": engine,
        })
    return public, hidden


def build_writing_taxonomy() -> dict[str, Any]:
    return {
        "description": "CN-GongWen-Writing：按目标产出 token 分桶的公文写作测试 prompt",
        "length_buckets": {
            k: {"label": v["label"], "target_tokens": list(v["target_tokens"]), "min_sections": v["min_sections"]}
            for k, v in LENGTH_BUCKETS.items()
        },
        "doc_types": [dt.name for dt in DOC_TYPES],
        "standards": list(STANDARDS),
        "framework_dimensions": ["标题三要素", "层次序数(一、（一）1.)", "行文方向规则", "签发人/主送/附注",
                                 "附件/抄送", "可执行性(责任-时限-反馈)", "结尾用语", "署名与成文日期"],
        "scored_dimensions": ["length", "title", "structure", "closing", "signatory", "recipient",
                              "directional", "executability", "punctuation", "language_safety", "trap_avoidance"],
        "forbidden_trap_pool": list(TRAP_TYPES),
        "forbidden_phrases": list(FORBIDDEN_PHRASES),
        "self_check": list(SELF_CHECK),
    }


def write_writing_dataset(count: int, use_llm: bool = False, cfg: LiteLLMConfig | None = None) -> dict[str, Any]:
    public, hidden = build_writing_dataset(count, use_llm, cfg)
    write_jsonl(DATASET3 / "writing_prompts_public.jsonl", public)
    write_jsonl(DATASET3 / "writing_prompts_with_rubric.jsonl", hidden)
    (DATASET3 / "taxonomy.json").write_text(
        json.dumps(build_writing_taxonomy(), ensure_ascii=False, indent=2), encoding="utf-8")
    buckets = {k: sum(1 for h in hidden if h["length_bucket"] == k) for k in LENGTH_ORDER}
    return {
        "writing_count": len(public),
        "writing_buckets": buckets,
        "writing_doc_types": len({h["spec"]["doc_type"] for h in hidden}),
        "writing_medical_share": round(sum(h["spec"]["policy_domain"] == "医疗卫生" for h in hidden) / max(1, len(hidden)), 3),
        "writing_prompt_engine": "minimax" if (use_llm and cfg is not None) else "deterministic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成公文写作测试 prompt（按目标产出 token 分短/中/长）")
    parser.add_argument("--count", type=int, default=90, help="题量（90 时恰覆盖 15 文种×3 长度×2 政策方向）")
    parser.add_argument("--use-litellm", action="store_true", help="用 LLM 一次 10 条撰写 prompt（需 MINIMAX_API_KEY）")
    args = parser.parse_args()
    use_llm = args.use_litellm and bool(os.getenv("MINIMAX_API_KEY"))
    cfg = LiteLLMConfig() if use_llm else None
    summary = write_writing_dataset(args.count, use_llm, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
