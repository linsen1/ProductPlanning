"""
AI 策划智能体 - 配置管理
一期聚焦：产品策划方案
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# 显式指定 .env 路径，确保无论从哪个目录启动 Streamlit 都能正确加载
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip().strip('"').strip("'")
    value = re.sub(r"\s+#.*$", "", value).strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = _env_value(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    value = _env_value(name, "1" if default else "0").lower()
    return value not in ("0", "false", "no", "off")

# ── 模型配置 ──────────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = _env_value("DASHSCOPE_API_KEY")
QWEN_MODEL        = _env_value("QWEN_MODEL", "qwen-plus")
EMBEDDING_MODEL   = _env_value("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_BACKEND = _env_value("EMBEDDING_BACKEND", "dashscope")
QWEN_BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"

OPENAI_API_KEY = _env_value("OPENAI_API_KEY")
OPENAI_MODEL   = _env_value("OPENAI_MODEL", "gpt-4o")

LLM_BACKEND = _env_value("LLM_BACKEND", "qwen")

# ── 博查搜索配置 ──────────────────────────────────────────────────────────────
# 博查全部从 .env 读取，代码中不再硬编码 key
BOCHA_API_KEY = _env_value("BOCHA_API_KEY")
BOCHA_API_URL = _env_value("BOCHA_API_URL", "https://api.bochaai.com/v1/web-search")
BOCHA_AISEARCH_URL = _env_value("BOCHA_AISEARCH_URL", "https://api.bochaai.com/v1/ai-search")
BOCHA_SEARCH_COUNT = _env_int("BOCHA_SEARCH_COUNT", 10)  # 每次搜索返回结果数（博查上限50）
BOCHA_FRESHNESS = _env_value("BOCHA_FRESHNESS", "noLimit")  # oneDay/oneWeek/oneMonth/oneYear/noLimit
# 竞品分析是否走 AI 搜索（返回综合长答案，深度更高）；其它检索仍走 web-search
BOCHA_USE_AISEARCH = _env_bool("BOCHA_USE_AISEARCH")

# ── 知识库配置 ────────────────────────────────────────────────────────────────
DB_DIR = _env_value("VECTOR_DB_DIR", "./chroma_db")
VECTOR_DB_MODE = _env_value("VECTOR_DB_MODE", "auto")
KB_DIR = _env_value("KB_DIR", "./data")
TOP_K  = _env_int("TOP_K", 3)

# ── 一期：项目类型（对应模版人群分类维度）────────────────────────────────────
PROJECT_TYPES = [
    "长考类 — 建筑工程（一建/二建/造价师等）",
    "长考类 — 医疗卫生（执医/护师/卫生资格等）",
    "长考类 — 财会税务（注会/税务师/初中级会计等）",
    "长考类 — 消防安全（一消/二消等）",
    "短考类 — 营养健康（营养师/健康管理师等）",
    "短考类 — 技能认证（育婴师/家政/养老等）",
    "技能类 — 数字技术（AIGC/PLC/自动化等）",
    "公考类 — 笔试（行测/申论）",
    "公考类 — 面试（结构化/无领导）",
    "党校/在职研究生",
    "其他",
]

# ── 人群学习动机维度（辅助填写引导）─────────────────────────────────────────
LEARNING_MOTIVATIONS = [
    "职场刚需（岗位晋升/持证上岗）",
    "求职转行（进入新行业/换岗）",
    "增收需求（副业接单/提高薪资）",
    "政策福利（补贴/积分落户等）",
    "自用需求（家庭/个人兴趣）",
]

# ── 价格定位区间 ──────────────────────────────────────────────────────────────
PRICE_TIERS = [
    "经济型（500元以内）",
    "主流型（500-2000元）",
    "高端型（2000-5000元）",
    "旗舰型（5000元以上）",
    "待定",
]

# ── 产品优势维度（对应模版"产品优势"章节）───────────────────────────────────
ADVANTAGE_DIMS = [
    "通过率保障",
    "师资力量",
    "教学体系/课程设计",
    "学习技术/产品体验",
    "服务保障",
    "退费/协议保障",
    "就业资源",
]

# ── P2：项目类型显示名 → 知识库 front-matter 归一化键 ────────────────────────
# 用于检索硬过滤：把 PROJECT_TYPES 的下拉显示值映射到 data/**/*.md 里的 project_type 键
PROJECT_TYPE_KEY = {
    "长考类 — 建筑工程（一建/二建/造价师等）": "长考类-建筑工程",
    "长考类 — 医疗卫生（执医/护师/卫生资格等）": "长考类-医疗卫生",
    "长考类 — 财会税务（注会/税务师/初中级会计等）": "长考类-财会税务",
    "长考类 — 消防安全（一消/二消等）": "长考类-消防安全",
    "短考类 — 营养健康（营养师/健康管理师等）": "短考类-营养健康",
    "短考类 — 技能认证（育婴师/家政/养老等）": "短考类-技能认证",
    "技能类 — 数字技术（AIGC/PLC/自动化等）": "技能类-数字技术",
    "公考类 — 笔试（行测/申论）": "公考类-笔试",
    "公考类 — 面试（结构化/无领导）": "公考类-面试",
    "党校/在职研究生": "党校在职研究生",
    "其他": "通用",
}

def project_type_key(display: str) -> str:
    """把项目类型显示名转为归一化键；未知则归一化前缀或返回'通用'"""
    if display in PROJECT_TYPE_KEY:
        return PROJECT_TYPE_KEY[display]
    # 兜底：取 "—" 前后拼接
    if "—" in display:
        a, b = display.split("—", 1)
        b = b.split("（")[0].strip()
        return f"{a.strip()}-{b}"
    return "通用"

# ── P2：方案模式 ──────────────────────────────────────────────────────────────
PLAN_MODES = ["单产品方案", "产品体系方案（多证书/多班型）"]

# ── P2：各项目类型下可选证书（产品体系模式用，驱动多张证书事实卡）──────────────
# 仅已沉淀知识库样板的证书给出明细，其余留空让策划人自填
CERTS_BY_TYPE = {
    "短考类-营养健康": ["公共营养师", "心理咨询师", "健康管理师"],
    "短考类-技能认证": ["育婴师", "保育师", "养老护理员"],
    "长考类-医疗卫生": ["执业医师", "执业护师", "卫生资格"],
}

# ── P2：分层软召回配额（按 layer 取多少片段进 prompt）──────────────────────────
SOFT_RECALL_QUOTAS = {
    "skill_modules": 4,   # 技能课是厚度来源，多取
    "career_paths":  2,
    "copy_bank":     2,
    "case":          2,
}


def validate_config() -> tuple[bool, str]:
    if LLM_BACKEND == "qwen" and not DASHSCOPE_API_KEY:
        return False, "未配置 DASHSCOPE_API_KEY，请在 .env 文件中配置"
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        return False, "未配置 OPENAI_API_KEY，请在 .env 文件中配置"
    return True, "OK"

def bocha_enabled() -> bool:
    return bool(BOCHA_API_KEY)
