# AI 策划智能体 · ProjectAI

> 基于 LangGraph + RAG + 千问3，面向项目运营策划部的产品策划方案智能生成工具

## 一期功能

填写产品立项信息 → AI 检索历史优秀案例 → 按标准模版生成完整《产品策划方案》：

| 章节 | 内容 |
|------|------|
| 🏷️ 主卖点 Slogan | 3个候选，≤12字，直白押韵 |
| 📖 产品背景 | 市场层 → 用户层 → 产品层，150-200字 |
| 👥 适合人群 | 身份角色/学习动机/学习门槛/应用场景 |
| 💪 产品优势 | 通过率/师资/教学/技术/服务/差异化 |
| 📑 产品简章 | 课程模块、配套资料、服务保障和价格 |
| 🎓 师资介绍 | 核心师资、授课特色和待确认信息 |
| 📚 课程说明 | 阶段课程、课时、上课形式和上线时间 |
| 💬 产介话术 | 朋友圈文案×3 + 1v1私聊话术 + 社群推送 |
| 🏗️ 产品搭建整体思路 | 卖点主轴、课程体系、服务配套、价格策略和推广切入点 |

## 快速启动

```bash
# 1. 进入项目目录
cd ProjectAI

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 4. 启动
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 云端部署

如果部署到 Streamlit Community Cloud，建议在 **Advanced settings → Python version** 选择 `3.12`，入口文件选择 `app.py`。如果已经创建了应用但误选了 Python 3.14，需要删除应用后重新部署，部署时再选择 Python 3.12。

Secrets / 环境变量至少需要配置：

```toml
LLM_BACKEND = "qwen"
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
BOCHA_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
QWEN_MODEL = "qwen3-235b-a22b"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_BACKEND = "local"
```

如果 DashScope 账号已开通 embedding 模型权限，可以把 `EMBEDDING_BACKEND` 改成 `"dashscope"`；未开通时保持 `"local"`，系统会使用本地关键词向量重建知识库。

## 知识库管理

知识库已按 7 层重构（详见 `data/README_知识库结构.md`），所有 `.md` 进入 RAG：

```
data/
├── 00_structure/     结构模板（怎么写）— template_00~05
├── 01_cert_facts/    证书事实卡 — 每证书：发证/级别/报考/考试/价格/合规
├── 02_skill_modules/ 实操技能课库 — 每证书技能模块清单
├── 03_career_paths/  就业变现库 — 线上/线下/全职三赛道
├── 04_copy_bank/     金句话术库 — 优势金句/Slogan/朋友圈/异议
├── 05_cases/         优秀成稿范例 — case_*
└── 06_compliance/    合规边界库（全局红线，优先级最高）
```

每个文件需带 YAML front-matter（`project_type/cert/layer/priority/applies_to`），供检索硬过滤。当前 01-04 层已做满"公共营养师/心理咨询师/健康管理师"三证书样板。

更新案例后，系统会根据 `data/**/*.md` 的文件清单自动重建索引；也可以点击侧边栏「🔄 重建知识库索引」手动刷新。

**案例模版格式参考：**
每个案例应包含：产品基本信息 + Slogan + 产品背景 + 适合人群 + 产品优势 + 产品简章 + 师资介绍 + 课程说明 + 产介话术 + 产品搭建整体思路，与《产品方案模版.docx》结构对齐。

**项目知识库模板格式参考：**
来自销售助手或项目部资料的内容，建议按“稳定事实层 + 销售适配层 + 客户可读层 + 更新风险层”整理，再映射到产品方案九章。模板里不要放原始附件路径、OCR 过程、覆盖审计或测试反馈；这些内容应保留在项目知识库自己的 `workbench/` 中。

## 技术架构

```
用户填写表单（产品名/项目类型/人群/优势/价格...）
    ↓
LangGraph 编排层
├── build_query（构建检索语句）
├── retrieve_kb（RAG检索历史案例 → Chroma 本地向量库）
└── generate_plan（Agno Agent + 千问3 生成方案）
    ↓
Streamlit 展示（分章节多 Tab 浏览）+ Word/Markdown 下载
```

## 文件结构

```
ProjectAI/
├── app.py              # 主应用（LangGraph工作流 + Streamlit UI）
├── config.py           # 配置管理（项目类型/动机/价格/优势维度）
├── prompts.py          # 提示词（PRODUCT_PLAN_PROMPT + 辅助函数）
├── requirements.txt    # 依赖清单
├── .env.example        # 环境变量模板
├── .streamlit/
│   └── config.toml     # UI主题配置
└── data/               # 知识库案例文件（.md格式）
    ├── case_01_建筑工程类产品策划.md
    ├── case_02_医疗卫生类产品策划.md
    └── case_03_数字技能类产品策划.md
```

## 迭代规划

| 阶段 | 内容 | 时间 |
|------|------|------|
| 一期 ✅ | 产品策划方案生成（当前） | 2026.06 |
| 二期 | 营销推广方案（活动策划/渠道投放/节点规划） | 待定 |
| 三期 | 执行清单生成（时间轴/责任人/物料/验收标准） | 待定 |

**一期后续优化：**
- [ ] 支持上传 Word/PDF 案例文件自动入库
- [ ] 增加方案历史记录和一键复用功能
- [ ] 团队协作：多人案例共享知识库（接入 WeKnora）
- [ ] 方案评分机制：AI 自检方案质量并给出优化建议
