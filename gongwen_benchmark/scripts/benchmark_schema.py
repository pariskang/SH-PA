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
    "doctype_misuse",       # 文种误用（请示/报告、函/请示、公告/通告、方案等非文种当文种）
    "addressing_relation",  # 行文关系（主送领导个人、越级未抄送、上行文抄送下级、原文转报）
    "authority_boundary",   # 权限边界（部门越权、未会签、内设机构对外行文、代党委政府决定）
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
    "policy_domain_classification", # 政策领域/医疗子领域分类
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


# --- 政策领域：通用政务 vs 医疗卫生（基准要求约各占一半）---
POLICY_DOMAINS = ("通用政务", "医疗卫生")


@dataclass(frozen=True)
class MedicalAreaSpec:
    code: str
    name: str
    description: str
    topics: tuple[str, ...]   # 该子领域下十分具体的政策主题分类


# 16 个医疗卫生政策子领域，每个再细分为若干十分具体的政策主题（合计约 105 个具体分类），
# 覆盖医保、医药、医疗、公卫、基层、中医药、监管、质量、妇幼、医养、健康促进、人才、职业健康、数智等各方面。
MEDICAL_AREAS: tuple[MedicalAreaSpec, ...] = (
    MedicalAreaSpec("MA01", "医保管理", "医保支付改革、目录、基金监管、参保与待遇", (
        "DRG按疾病诊断相关分组付费改革", "DIP按病种分值付费改革", "基本医疗保险药品目录管理",
        "医保基金使用常态化监管", "职工基本医保门诊共济保障", "城乡居民基本医保参保扩面",
        "异地就医住院与门诊直接结算", "长期护理保险制度建设", "生育保险待遇保障",
    )),
    MedicalAreaSpec("MA02", "医药供应与集采", "集采、国谈药、短缺药、基本药物", (
        "药品集中带量采购落地执行", "高值医用耗材集中带量采购", "国家谈判药品落地与“双通道”管理",
        "短缺药品保供稳价", "国家基本药物制度巩固完善", "药品价格监测与成本调查",
    )),
    MedicalAreaSpec("MA03", "医疗服务价格", "价格调整、价格项目、备案", (
        "医疗服务价格动态调整", "检查检验项目价格规范管理", "医疗服务价格项目立项编制",
        "公立医院医疗服务价格备案",
    )),
    MedicalAreaSpec("MA04", "公立医院改革", "高质量发展、绩效、薪酬、治理", (
        "公立医院高质量发展", "公立医院绩效国家监测考核", "公立医院薪酬制度改革",
        "现代医院管理制度与医院章程建设", "公立医院全面预算与运营管理", "公立医院党建引领发展",
    )),
    MedicalAreaSpec("MA05", "分级诊疗", "医联体、医共体、签约、转诊", (
        "紧密型城市医疗集团建设", "紧密型县域医共体建设", "家庭医生签约服务",
        "医疗机构双向转诊", "基层首诊与连续服务", "专科联盟与远程医疗协作网建设",
    )),
    MedicalAreaSpec("MA06", "公共卫生", "传染病、疾控、免疫、应急、慢病", (
        "重大传染病疫情防控", "疾病预防控制体系改革", "国家免疫规划疫苗接种",
        "突发公共卫生事件应急处置", "慢性非传染性疾病综合防控", "地方病防治",
        "严重精神障碍患者服务管理", "病媒生物预防控制",
    )),
    MedicalAreaSpec("MA07", "基层卫生", "社区、卫生院、村室、基本公卫", (
        "社区卫生服务中心标准化建设", "乡镇卫生院服务能力提升", "村卫生室规范化管理",
        "国家基本公共卫生服务项目落实", "家庭医生团队建设", "基层卫生人才“县管乡用”",
    )),
    MedicalAreaSpec("MA08", "中医药", "传承创新、机构、中西医结合、中药", (
        "中医药传承创新发展", "中医医疗机构服务能力建设", "中西医结合临床协作",
        "中药饮片质量管理", "中医药适宜技术推广", "基层中医馆（国医堂）建设",
    )),
    MedicalAreaSpec("MA09", "药品器械监管", "药品器械、疫苗、药物警戒、特药", (
        "药品生产经营质量监管", "医疗器械注册与使用监管", "疫苗全程追溯管理",
        "药物警戒与药品不良反应监测", "麻醉药品和精神药品管理", "药品网络销售监督", "化妆品安全监管",
    )),
    MedicalAreaSpec("MA10", "医疗质量与安全", "质控、院感、用血、用药、纠纷", (
        "医疗质量管理与控制", "医疗机构感染预防与控制", "临床用血管理",
        "合理用药与处方点评", "抗菌药物临床应用管理", "医疗纠纷预防和处理",
        "手术安全核查与患者安全", "病案首页与病案质量管理",
    )),
    MedicalAreaSpec("MA11", "妇幼健康", "母婴、出生缺陷、近视、托育", (
        "母婴安全保障", "出生缺陷综合防治", "儿童青少年近视防控",
        "普惠托育服务体系建设", "适龄妇女“两癌”筛查", "婚前与孕前保健",
    )),
    MedicalAreaSpec("MA12", "老龄与医养结合", "医养、安宁疗护、老年健康", (
        "医养结合服务发展", "安宁疗护服务", "老年健康服务体系建设",
        "失能老年人长期照护", "老年友善医疗机构建设", "老年认知障碍（痴呆）防治",
    )),
    MedicalAreaSpec("MA13", "健康促进", "健康中国、控烟、营养、心理", (
        "健康中国行动实施", "无烟环境建设与控烟", "全民健身与全民健康融合",
        "国民营养计划与合理膳食", "社会心理服务体系建设", "健康县区与健康城市建设", "居民健康素养提升",
    )),
    MedicalAreaSpec("MA14", "卫生人才与教育", "住培、全科、执业、职称", (
        "住院医师规范化培训", "全科医生培养使用", "医师执业注册管理",
        "护士队伍建设", "医学继续教育", "卫生专业技术人员职称评审", "儿科精神科麻醉等紧缺人才培养",
    )),
    MedicalAreaSpec("MA15", "职业健康", "职业病、监护、尘肺、放射卫生", (
        "职业病防治", "用人单位职业健康监护", "用人单位职业卫生主体责任落实",
        "尘肺病防治攻坚", "职业卫生分类监督执法", "放射卫生防护监管",
    )),
    MedicalAreaSpec("MA16", "互联网医疗与数据", "互联网医院、远程、健康档案、数据", (
        "互联网医院与互联网诊疗管理", "远程医疗协同网络建设", "居民电子健康档案规范应用",
        "医疗健康大数据安全与共享", "智慧医院与电子病历分级评价", "检查检验结果互通互认", "电子处方流转管理",
    )),
)


def medical_area_by_name(name: str) -> MedicalAreaSpec:
    return next(spec for spec in MEDICAL_AREAS if spec.name == name)


def all_medical_topics() -> tuple[str, ...]:
    return tuple(topic for area in MEDICAL_AREAS for topic in area.topics)


def doc_type_by_code(code: str) -> DocTypeSpec:
    return next(spec for spec in DOC_TYPES if spec.code == code)


def doc_type_by_name(name: str) -> DocTypeSpec:
    return next(spec for spec in DOC_TYPES if spec.name == name)


def element_by_code(code: str) -> ElementSpec:
    return next(spec for spec in ELEMENTS if spec.code == code)


# --- 权威依据：公文写作四类规范 ---
STANDARDS = (
    "《党政机关公文处理工作条例》(2012)",
    "GB/T 9704—2012《党政机关公文格式》",
    "GB/T 15834—2011《标点符号用法》",
    "GB/T 15835—2011《出版物上数字用法》",
)

# --- 公文语言“安全写法”：规范表达 vs 须规避的夸大/网络化/情绪化表达 ---
PREFERRED_PHRASES = (
    "根据", "按照", "为贯彻落实", "经研究", "现将有关事项通知如下",
    "请结合实际认真贯彻执行", "请于规定时限前报送",
)
FORBIDDEN_PHRASES = (
    "史无前例", "极其重要地", "全方位赋能", "打造最强生态", "颠覆式创新",
    "绝对领先", "完美闭环", "全面解决所有问题", "遥遥领先", "震撼发布",
)

# --- 部署性下行文须具备的“可执行性”要素（责任主体+完成时限+报送反馈，避免只有口号）---
EXECUTABLE_DOC_TYPES = ("通知", "意见", "决定")

# --- 成稿前“十项硬核自查” ---
SELF_CHECK = (
    "文种对不对：是否属于15种法定公文，未把申请/说明/方案误作文种",
    "主送对不对：上行/下行/平行关系准确，无多头请示、越级、主送领导个人",
    "权限对不对：本单位有权发文，必要时经上级授权、部门会签或会议审议",
    "依据对不对：政策法规、上级文件、会议决定、领导批示真实有效完整",
    "事项清不清：目标、任务、责任、时限、标准、反馈方式明确",
    "数据准不准：人名地名单位名文号日期金额比例附件序号前后一致",
    "口径稳不稳：无过度承诺、越权表态、绝对化表达、涉密或敏感内容",
    "格式合不合：标题正文层级序号附件落款日期印章页码符合规范",
    "标点对不对：发文字号用六角括号〔〕，层级序号规范，附件说明正确",
    "流程齐不齐：经部门核稿、办公室审核、合法性审查、领导签发、会签、登记、归档",
)


@dataclass(frozen=True)
class DocTypeGuide:
    name: str
    usage: str                     # 主要用途
    key_points: tuple[str, ...]    # 写作要点
    pitfalls: tuple[str, ...]      # 典型雷区（自然语言）


# 文种辨析（主要用途 / 写作要点 / 典型雷区）——重点覆盖请示·报告·函等高频易错文种
DOC_TYPE_GUIDE: dict[str, DocTypeGuide] = {
    "通知": DocTypeGuide(
        "通知", "发布、传达、部署、转发、批转事项",
        ("写清依据、事项、对象、要求、时限、联系人", "部署性通知须明确责任单位与完成时限"),
        ("把通知写成泛泛倡议", "任务无责任单位和时限")),
    "请示": DocTypeGuide(
        "请示", "请求上级指示、批准",
        ("一文一事；理由充分；提出明确请示事项；结尾常用“妥否，请批示”", "只主送一个上级机关、须标签发人、附注注明联系人电话"),
        ("多头请示", "报告夹带请示", "请示事项含混")),
    "报告": DocTypeGuide(
        "报告", "汇报工作、反映情况、回复询问",
        ("事实准确；问题客观；建议可写但不能要求批复",),
        ("写成请示", "夹带“请予批准”")),
    "函": DocTypeGuide(
        "函", "不相隶属机关商洽、询问、答复、请求批准",
        ("语气平和；事项具体；来往关系清楚",),
        ("平级单位却写“请示”", "语气过硬或越权")),
    "通报": DocTypeGuide(
        "通报", "表彰先进、批评错误、传达精神、告知情况",
        ("事实清楚；评价准确；导向鲜明",),
        ("事实未经核实", "批评性通报措辞过重")),
    "批复": DocTypeGuide(
        "批复", "答复下级请示事项",
        ("针对请示逐项答复；态度明确；依据充分",),
        ("未对应请示事项", "答复含糊")),
    "意见": DocTypeGuide(
        "意见", "对重要问题提出见解和处理办法",
        ("政策性、指导性强；措施成体系",),
        ("写成工作方案但缺政策依据",)),
    "纪要": DocTypeGuide(
        "纪要", "记载会议主要情况和议定事项",
        ("写会议议定事项，不写流水账",),
        ("用纪要替代正式审批、处罚、任免或行政决定",)),
}


def doc_type_guide(name: str) -> DocTypeGuide | None:
    return DOC_TYPE_GUIDE.get(name)


# --- 医疗卫生公文的权威依据（含 2024–2026 新规，知识截至 2026-06-24）---
MEDICAL_STANDARDS = (
    "《基本医疗卫生与健康促进法》",
    "《医疗机构管理条例》",
    "《医师法》",
    "《医疗质量管理办法》与《医疗质量安全核心制度要点》",
    "《医疗机构病历管理规定》",
    "《医疗纠纷预防和处理条例》",
    "《个人信息保护法》《数据安全法》",
    "《传染病防治法》（2025年9月1日施行）",
    "《涉及人的生命科学和医学研究伦理审查办法》",
    "《医疗卫生机构开展研究者发起的临床研究管理办法》（2024年10月1日施行）",
    "《医疗保障基金使用监督管理条例实施细则》（2026年4月1日施行）",
    "《医疗广告管理办法》《广告法》",
)

# 八条医疗合规线
MEDICAL_COMPLIANCE_LINES = (
    "依法合规线", "患者安全线", "医学证据线", "患者隐私和数据安全线",
    "知情同意线", "伦理审查线", "公共卫生报告线", "医保基金合规线",
)

# 医疗公文须规避的高风险表述（疗效夸大 / 绝对化 / 广告红线 / AI 越界）
MEDICAL_FORBIDDEN_PHRASES = (
    "确保治愈", "彻底治愈", "根治", "包治", "无副作用", "一次见效", "治愈率100%",
    "患者满意率100%", "完全避免并发症", "疗效领先", "全国第一", "顶级专家",
    "保证治好", "自动诊断", "无需医生审核", "替代医师",
)

# 医疗公文推荐的稳妥表述（与上面构成正反对照）
MEDICAL_PREFERRED_PHRASES = (
    "有助于提升", "初步显示改善趋势", "在符合适应证和规范诊疗前提下", "仍需进一步临床研究验证",
    "作为临床辅助决策工具", "须由具备资质的医务人员审核确认", "已进行必要去标识化处理",
)

# 医疗公文成稿前 12 项高风险审核清单
MEDICAL_SELF_CHECK = (
    "文种是否正确：通知/报告/请示/函/通报/纪要是否混用",
    "发文权限是否匹配：科室/医务部/医院/卫健委/医保局/疾控权限是否越界",
    "法律依据是否最新：2025传染病防治法、2024 IIT 管理办法、2026 医保基金监管实施细则",
    "医学事实是否准确：诊断/术式/药品/器械/指标/病例数/死亡数/感染数是否核对",
    "是否夸大疗效：有无治愈/根治/无风险/保证/最优/唯一/领先等绝对化词",
    "是否泄露隐私：附件/截图/病历/影像/基因/罕见病信息是否可识别患者",
    "是否完成伦理审查：涉人样本/数据/临床研究/AI 训练数据是否有伦理与授权依据",
    "是否符合知情同意：手术/特殊检查治疗/临床试验是否说明风险与替代方案",
    "是否符合核心制度：首诊负责/三级查房/会诊/危急值/手术分级/病历/抗菌药物",
    "是否存在医保风险：收费/编码/病案首页/处方/耗材/DRG-DIP 是否触发基金监管",
    "是否涉及公共卫生报告权限：传染病/院感暴发/突发公卫事件是否按规定上报发布",
    "是否留痕可追溯：数据来源/纪要/审批/会签/整改台账/附件版本是否完整",
)
