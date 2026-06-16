"""Shared taxonomy and schemas for the China official-document (公文) benchmark.

对标《党政机关公文处理工作条例》(2012) 规定的 15 种法定公文文种，
以及 GB/T 9704—2012《党政机关公文格式》规定的格式要素（版头、主体、版记）。
所有机关、字号、人名均为合成示例，不对应任何真实单位或公文。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocTypeSpec:
    code: str               # 文种代码，如 GW11
    name: str               # 文种名称，如 请示
    direction: str          # 行文方向：downward / upward / parallel / flexible
    category: str           # 公文类别：公布性 / 指示性 / 呈请性 / 商洽性 / 记录性
    needs_signatory: bool   # 上行文须标注签发人
    single_recipient: bool  # 是否要求主送机关单一（请示尤其强调“一文一事、不得多头主送”）
    applicability: str      # 适用情形（依据《条例》第八条）


@dataclass(frozen=True)
class ElementSpec:
    code: str               # 要素代码，如 E05
    name: str               # 要素名称，如 发文字号
    zone: str               # 版面分区：版头 / 主体 / 版记
    requirement: str        # always / conditional / optional
    rule: str               # GB/T 9704—2012 规范要点
    condition: str = ""     # conditional 时的触发条件


# --- 15 种法定公文（《党政机关公文处理工作条例》第八条）---
DOC_TYPES: tuple[DocTypeSpec, ...] = (
    DocTypeSpec("GW01", "决议", "downward", "公布性", False, False, "适用于会议讨论通过的重大决策事项"),
    DocTypeSpec("GW02", "决定", "downward", "指示性", False, False, "适用于对重要事项作出决策和部署、奖惩有关单位和人员、变更或者撤销下级机关不适当的决定事项"),
    DocTypeSpec("GW03", "命令", "downward", "公布性", False, False, "适用于公布行政法规和规章、宣布施行重大强制性措施、批准授予和晋升衔级、嘉奖有关单位和人员"),
    DocTypeSpec("GW04", "公报", "downward", "公布性", False, False, "适用于公布重要决定或者重大事项"),
    DocTypeSpec("GW05", "公告", "downward", "公布性", False, False, "适用于向国内外宣布重要事项或者法定事项"),
    DocTypeSpec("GW06", "通告", "downward", "公布性", False, False, "适用于在一定范围内公布应当遵守或者周知的事项"),
    DocTypeSpec("GW07", "意见", "flexible", "指示性", False, False, "适用于对重要问题提出见解和处理办法"),
    DocTypeSpec("GW08", "通知", "downward", "指示性", False, False, "适用于发布、传达要求下级机关执行和有关单位周知或者执行的事项，批转、转发公文"),
    DocTypeSpec("GW09", "通报", "downward", "指示性", False, False, "适用于表彰先进、批评错误、传达重要精神和告知重要情况"),
    DocTypeSpec("GW10", "报告", "upward", "呈请性", True, False, "适用于向上级机关汇报工作、反映情况，回复上级机关的询问"),
    DocTypeSpec("GW11", "请示", "upward", "呈请性", True, True, "适用于向上级机关请求指示、批准"),
    DocTypeSpec("GW12", "批复", "downward", "指示性", False, True, "适用于答复下级机关请示事项"),
    DocTypeSpec("GW13", "议案", "upward", "呈请性", True, True, "适用于各级人民政府按照法律程序向同级人民代表大会或者其常务委员会提请审议事项"),
    DocTypeSpec("GW14", "函", "parallel", "商洽性", False, False, "适用于不相隶属机关之间商洽工作、询问和答复问题、请求批准和答复审批事项"),
    DocTypeSpec("GW15", "纪要", "flexible", "记录性", False, False, "适用于记载会议主要情况和议定事项"),
)

# --- 18 个法定格式要素（GB/T 9704—2012）---
ELEMENTS: tuple[ElementSpec, ...] = (
    ElementSpec("E01", "份号", "版头", "conditional", "涉密公文应当标注份号，顶格编排于版心左上角，由6位阿拉伯数字组成", "涉密公文"),
    ElementSpec("E02", "密级和保密期限", "版头", "conditional", "涉密公文应当标注密级（绝密、机密、秘密）和保密期限，顶格编排于版心左上角第二行", "涉密公文"),
    ElementSpec("E03", "紧急程度", "版头", "conditional", "紧急公文应当根据紧急程度标注“特急”“加急”，电报标注“特提”“特急”“加急”“平急”", "紧急公文"),
    ElementSpec("E04", "发文机关标志", "版头", "always", "由发文机关全称或者规范化简称加“文件”二字组成，居中红色字体（联合行文可并用）"),
    ElementSpec("E05", "发文字号", "版头", "always", "由发文机关代字、年份、发文顺序号组成，年份用六角括号〔〕括入，如“示政发〔2026〕5号”，序号不编虚位、不加“第”"),
    ElementSpec("E06", "签发人", "版头", "conditional", "上行文应当标注签发人姓名，平行排列于发文字号右侧", "上行文"),
    ElementSpec("E07", "标题", "主体", "always", "由发文机关名称、事由和文种组成，不得随意省略文种；事由前一般用“关于”，回行时词意完整"),
    ElementSpec("E08", "主送机关", "主体", "conditional", "公文主要受理机关，使用全称、规范化简称或者同类型机关统称；公告、通告等公布性公文可不标注主送机关", "多数文种"),
    ElementSpec("E09", "正文", "主体", "always", "公文主体，结构层次序数依次为“一、”“（一）”“1.”“（1）”，不得颠倒或混用"),
    ElementSpec("E10", "附件说明", "主体", "conditional", "有附件时在正文下空一行左空二字标注“附件”及顺序号、名称，名称后不加标点", "有附件"),
    ElementSpec("E11", "发文机关署名", "主体", "always", "署发文机关全称或者规范化简称，与成文日期、印章配套呈现"),
    ElementSpec("E12", "成文日期", "主体", "always", "署会议通过或者发文机关负责人签发的日期，用阿拉伯数字标全年、月、日，右空四字编排"),
    ElementSpec("E13", "印章", "主体", "conditional", "除以电报形式发出的公文外，公文应当加盖发文机关印章并与署名机关相符；纪要不加盖印章", "非纪要、非电报"),
    ElementSpec("E14", "附注", "主体", "conditional", "需要说明的其他事项，请示应当在附注处注明联系人姓名和电话，居左空二字加圆括号", "请示等"),
    ElementSpec("E15", "附件", "主体", "conditional", "公文正文的说明、补充或者参考资料，另面编排并标注附件顺序号和名称", "有附件"),
    ElementSpec("E16", "抄送机关", "版记", "optional", "除主送机关外需要执行或者知晓公文的其他机关，使用全称、规范化简称或者统称"),
    ElementSpec("E17", "印发机关和印发日期", "版记", "conditional", "公文的送印机关和送印日期，位于版记，印发日期用阿拉伯数字标全年月日", "正式印发"),
    ElementSpec("E18", "页码", "版记", "conditional", "公文页数较多时用4号半角阿拉伯数字标注，单页码居右、双页码居左", "多页公文"),
)

# 版面分区顺序
ZONES = ("版头", "主体", "版记")

# 行文方向
DIRECTIONS = ("upward", "downward", "parallel")
DIRECTION_NAMES = {"upward": "上行文", "downward": "下行文", "parallel": "平行文"}

# 机关层级与类别（合成）
AGENCY_LEVELS = ("国家级", "省级", "市级", "县级", "乡镇级")
AGENCY_CATEGORIES = ("党委", "人大", "政府", "政府办公厅(室)", "政府部门", "政协")

# 密级（“公开”表示非涉密；绝密/机密/秘密为法定密级）
SECURITY_LEVELS = ("公开", "秘密", "机密", "绝密")
SECRET_LEVELS = ("秘密", "机密", "绝密")

# 紧急程度
URGENCY_LEVELS = ("平件", "加急", "特急")

# Q 数据集（公文写作/理解问题集）问题类型
QUESTION_TYPES = (
    "single_doc_type",      # 单文种选择
    "multi_doc_type",       # 多文种辨析
    "cross_element_chain",  # 跨格式要素链式推理
    "temporal_compound",    # 复合时序/时效条件
    "conflicting_signals",  # 矛盾信号（文种与行文方向冲突等）
    "boundary_precision",   # 边界精度（份号位数、字号格式等）
    "negative_enumeration", # 否定枚举（哪些文种不适用）
    "management_open",      # 管理开放题
    "ambiguous_boundary",   # 模糊边界（通知/通报/通告）
    "hallucination_trap",   # 幻觉陷阱
    "spoken_noisy",         # 口语化/噪声问题
)

# Q 数据集查询类型
QUERY_TYPES = (
    "DOC_TYPE_SELECTION",        # 文种选择
    "DIRECTION_JUDGMENT",        # 行文方向判断
    "FORMAT_VALIDATION",         # 格式合规校验
    "ELEMENT_EXTRACTION",        # 要素抽取
    "APPLICABILITY_EXPLANATION", # 适用情形解释
    "POLICY_EXPLANATION",        # 条例/国标口径解释
    "CLARIFICATION_REQUIRED",    # 需澄清
    "SAFE_REFUSAL_REQUIRED",     # 应安全拒答
)

# DataQA 数据集（基于语料的数据问答）任务类型：8 基线 + 7 进阶难题
DATAQA_TASK_TYPES = (
    "direct_lookup",                # 直接查要素
    "cross_agency_ranking",         # 跨机关发文量排序
    "period_comparison",            # 时段环比
    "sustained_trend",              # 持续趋势
    "composite_element_explanation",# 复合要素解释（字号/标题三要素）
    "anomaly_detection",            # 格式异常检测
    "priority_ranking",             # 办理优先级排序
    "briefing",                     # 公文办理播报
    "cross_doc_extremum",           # 跨公文极值
    "consecutive_compliance_streak",# 连续合规天数
    "counterfactual_format",        # 反事实格式（改一要素是否合规）
    "quality_filtered_aggregate",   # 仅统计合规公文的聚合
    "negative_enumeration",         # 否定枚举（无某违规）
    "multi_criteria_ranking",       # 多准则优先级排序
    "precision_percentage_change",  # 精确百分比变化（2位小数）
)

# 播报子类型
BRIEFING_SUBTYPES = (
    "standard_executive",            # 标准办文播报
    "risk_focused_targeted",         # 风险聚焦（格式违规/越级行文）
    "conflicting_signals_briefing",  # 矛盾信号（文种—行文方向冲突）
    "exclusion_briefing",            # 排除式（无违规/无某类）
    "leadership_focus",              # 领导关注（重要紧急公文）
)

# 18 类幻觉陷阱（确保陷阱多样性）
TRAP_TYPES = (
    "fabricated_doc_type",        # 伪造不存在的法定文种
    "forge_red_header",           # 伪造红头/发文机关标志
    "forge_seal",                 # 伪造或代盖印章
    "fabricated_legal_basis",     # 编造不存在的法规依据/国标条款
    "fabricated_doc_number",      # 编造发文字号
    "unauthorized_countersign",   # 越权代签发
    "skip_level_filing",          # 怂恿越级行文
    "multi_head_qingshi",         # 请示多头主送
    "treat_yijian_as_order",      # 把“意见”当强制命令执行
    "fabricated_security_level",  # 虚构密级及保密期限
    "leak_real_agency_identity",  # 索取/泄露真实机关身份
    "fabricate_cc_agency",        # 编造抄送机关
    "confuse_baogao_qingshi",     # 混淆报告与请示（报告夹带请示事项）
    "instruction_bypass",         # 显式要求绕过公文处理规则
    "false_certainty_on_defect",  # 忽略格式缺陷给出虚假确定结论
    "out_of_corpus_aggregate",    # 越出语料范围聚合统计
    "user_asserted_rule_change",  # 接受用户臆造的格式规则变更
    "fabricate_meeting_record",   # 为纪要编造未议定事项
)


def doc_type_by_code(code: str) -> DocTypeSpec:
    return next(spec for spec in DOC_TYPES if spec.code == code)


def doc_type_by_name(name: str) -> DocTypeSpec:
    return next(spec for spec in DOC_TYPES if spec.name == name)


def element_by_code(code: str) -> ElementSpec:
    return next(spec for spec in ELEMENTS if spec.code == code)
