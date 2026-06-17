"""
对话精修引擎 — ProjectAI
职责：
  1. 意图解析：分析用户修改请求（类型、范围、是否模糊）
  2. 修改执行：局部（指定章节）或全局（整体风格/结构）修改
  3. 版本管理：每次修改存档，支持回滚到任意版本
  4. 主动建议：初次进入精修模式时扫描方案，主动提出改进点
"""
from __future__ import annotations

import datetime
import json
import re
from typing import Optional

import streamlit as st


# ── Prompts ───────────────────────────────────────────────────────────────────

INTENT_PROMPT = """\
你是产品策划方案的精修助手。分析用户的修改请求，返回JSON。

用户请求：{user_message}

当前方案节选（前600字）：
{plan_preview}

分类规则：
- scope: "local"（只改某章节/段落）或 "global"（改整体风格/结构）
- type: "text"（话术/文字调整）| "structure"（增删/调整章节顺序）| "data"（价格/名称/日期等数据）| "style"（整体风格/语气）
- section: 涉及的章节名称关键词，如"产品背景""课程说明""招生简章"（local时必填，global时填null）
- has_ambiguity: 请求是否模糊需要澄清（true/false）
- clarification: 若模糊，需要问用户什么（否则填null）
- is_rollback: 用户是否在请求回滚历史版本（true/false）
- rollback_to: 若回滚，目标版本号整数（否则填null）
- is_finalize: 用户是否说定稿/完成/没问题了（true/false）
- proactive_note: 若方案有明显问题可主动提示，不超过35字（否则填null）

仅返回JSON，不要其他内容。"""


LOCAL_PROMPT = """\
你是优路教育产品策划专家，正在精修方案的指定章节。

用户请求：{user_message}

目标章节【{section}】当前内容：
{section_content}

上下文参考（前后章节标题，不要修改）：
{context_hint}

要求：
1. 只输出修改后的「{section}」章节内容（含章节标题行）
2. 若修改内容与行业规范/合规要求冲突，在章节末尾追加「💡 建议：...」一行说明
3. 不要输出其他章节的内容

仅返回该章节的Markdown内容，不要任何其他文字。"""


GLOBAL_PROMPT = """\
你是优路教育产品策划专家，正在对整份方案进行全局调整。

用户请求：{user_message}

当前完整方案：
{current_plan}

要求：
1. 按用户请求调整整份方案
2. 保持方案完整性，不遗漏任何章节
3. 若调整风格/语气，全篇统一调整，不要只改前半段
4. 在方案末尾追加一行：「✅ 本次调整：{summary}」

仅返回完整方案Markdown，不要任何其他文字。"""


SUGGEST_PROMPT = """\
你是优路教育产品策划质量审核专家。快速扫描下面这份方案，找2-3个最值得优化的问题点。

方案节选：
{plan_preview}

要求：
- 每条直接说「哪里有什么问题，建议怎么改」
- 每条不超过40字，以「• 」开头
- 重点关注：课程描述是否有价值说明、特色课是否突出、价格档位是否清晰、话术是否有说服力

仅返回问题列表，不要其他文字。"""


# ── Version Store ─────────────────────────────────────────────────────────────

class VersionStore:
    """在 st.session_state 里管理方案版本历史，支持无限轮修改和任意版本回滚"""

    KEY = "plan_versions"

    def __init__(self):
        if self.KEY not in st.session_state:
            st.session_state[self.KEY] = []

    def push(self, content: str, description: str) -> int:
        versions: list = st.session_state[self.KEY]
        num = len(versions) + 1
        versions.append({
            "num": num,
            "content": content,
            "description": description,
            "ts": datetime.datetime.now().strftime("%H:%M"),
        })
        return num

    def current(self) -> Optional[str]:
        v: list = st.session_state.get(self.KEY, [])
        return v[-1]["content"] if v else None

    def current_num(self) -> int:
        v: list = st.session_state.get(self.KEY, [])
        return v[-1]["num"] if v else 0

    def get(self, num: int) -> Optional[str]:
        for v in st.session_state.get(self.KEY, []):
            if v["num"] == num:
                return v["content"]
        return None

    def rollback(self, num: int) -> Optional[str]:
        """回滚到指定版本（保存为新版本，原历史不丢失）"""
        content = self.get(num)
        if content:
            self.push(content, f"回滚到第{num}版")
        return content

    def list_versions(self) -> list[dict]:
        return list(st.session_state.get(self.KEY, []))

    def count(self) -> int:
        return len(st.session_state.get(self.KEY, []))

    def clear(self):
        st.session_state[self.KEY] = []


# ── 意图解析 ──────────────────────────────────────────────────────────────────

def parse_intent(user_message: str, current_plan: str, llm) -> dict:
    """
    解析用户修改意图，返回结构化 intent dict。
    失败时返回安全兜底值（全局文字修改）。
    """
    prompt = INTENT_PROMPT.format(
        user_message=user_message,
        plan_preview=current_plan[:600],
    )
    default = {
        "scope": "global", "type": "text", "section": None,
        "has_ambiguity": False, "clarification": None,
        "is_rollback": False, "rollback_to": None,
        "is_finalize": False, "proactive_note": None,
    }
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", str(resp))
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {**default, **parsed}
    except Exception:
        pass
    return default


# ── 章节定位 ──────────────────────────────────────────────────────────────────

def _find_section_bounds(plan: str, section_keyword: str) -> tuple[int, int]:
    """
    在方案文本中定位章节边界，返回 (start_line, end_line) 行索引。
    找不到时返回 (-1, -1)（调用方据此判断，绝不可用整篇替换，否则会丢失其他章节）。

    匹配策略（从严到宽）：
      1) 标题文字与 keyword 完全相等（去掉 # 和首尾空白后）；
      2) keyword 作为子串出现在标题里（关键字已转义，避免正则注入）。
    """
    keyword = (section_keyword or "").strip()
    if not keyword:
        return -1, -1

    lines = plan.split("\n")
    start = -1
    level = 2
    safe_kw = re.escape(keyword)

    # 第一轮：标题文字精确相等
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s+(.*)$", line.strip())
        if m and m.group(2).strip() == keyword:
            start = i
            level = len(m.group(1))
            break

    # 第二轮：标题里包含关键字（转义后子串匹配）
    if start == -1:
        for i, line in enumerate(lines):
            m = re.match(r"^(#+)\s", line.strip())
            if m and re.search(safe_kw, line, re.IGNORECASE):
                start = i
                level = len(m.group(1))
                break

    if start == -1:
        return -1, -1

    end = len(lines)
    for i in range(start + 1, len(lines)):
        m = re.match(r"(#+)\s", lines[i])
        if m and len(m.group(1)) <= level:  # group(1)=纯#号，不含空白
            end = i
            break

    return start, end


# ── 修改执行 ──────────────────────────────────────────────────────────────────

def _splice_section(plan: str, s: int, e: int, new_section: str) -> str:
    """将方案的 [s, e) 行替换为 new_section，返回拼合后的完整方案。"""
    lines = plan.split("\n")
    new_lines = lines[:s] + new_section.split("\n") + lines[e:]
    return "\n".join(new_lines)


def apply_refinement(
    user_message: str,
    current_plan: str,
    intent: dict,
    llm,
) -> tuple[str, str]:
    """
    执行修改，返回 (新版本内容, 修改描述)。
    local 模式：LLM 只输出目标章节，代码拼回完整方案（大幅减少 token）。
    global 模式：LLM 输出完整方案。
    LLM 失败时返回原文 + 错误描述。
    """
    scope = intent.get("scope", "global")
    section = (intent.get("section") or "").strip()

    # 局部模式：先确认能定位到章节。定位不到则降级为全局，
    # 绝不可用「单章节输出」替换整篇 —— 那会把其他模块全部丢失。
    s = e = -1
    if scope == "local" and section:
        s, e = _find_section_bounds(current_plan, section)
        if s == -1:
            scope = "global"

    if scope == "local" and section and s != -1:
        lines = current_plan.split("\n")
        section_content = "\n".join(lines[s:e])

        # 构造上下文提示（前后章节标题，帮助模型理解位置）
        prev_titles = [l for l in lines[:s] if re.match(r"#+\s", l)][-2:]
        next_titles = [l for l in lines[e:] if re.match(r"#+\s", l)][:2]
        context_hint = "前：" + " | ".join(prev_titles) if prev_titles else ""
        context_hint += ("  后：" + " | ".join(next_titles)) if next_titles else ""

        prompt = LOCAL_PROMPT.format(
            user_message=user_message,
            section=section,
            section_content=section_content,
            context_hint=context_hint or "（首章节）",
        )
        desc = f"局部·{section[:10]}"

        try:
            resp = llm.invoke(prompt)
            new_section = getattr(resp, "content", str(resp)).strip()
            new_section = re.sub(r"^```\w*\n?", "", new_section)
            new_section = re.sub(r"\n?```$", "", new_section).strip()
            if not new_section:
                return current_plan, "修改失败：模型未返回内容"
            # 安全校验：拼回后不得整篇坍缩（防止极端情况丢失其他章节）
            new_plan = _splice_section(current_plan, s, e, new_section)
            if len(new_plan) < len(current_plan) * 0.5:
                # 章节修改不该让全文缩水一半以上，多半是模型把整篇当成单章节重写了
                return current_plan, "修改未生效：结果异常（已保留原文，请重试或缩小修改范围）"
            return new_plan, desc
        except Exception as exc:
            return current_plan, f"修改失败：{str(exc)[:40]}"

    else:
        summary = user_message[:25]
        prompt = GLOBAL_PROMPT.format(
            user_message=user_message,
            current_plan=current_plan,
            summary=summary,
        )
        desc = f"全局·{user_message[:15]}"

        try:
            resp = llm.invoke(prompt)
            new_content = getattr(resp, "content", str(resp)).strip()
            new_content = re.sub(r"^```\w*\n?", "", new_content)
            new_content = re.sub(r"\n?```$", "", new_content).strip()
            if not new_content:
                return current_plan, "修改失败：模型未返回内容"
            # 章节数 / 篇幅守卫：防止模型输出被截断而丢掉大量章节
            old_secs = len(re.findall(r"(?m)^##\s", current_plan))
            new_secs = len(re.findall(r"(?m)^##\s", new_content))
            if old_secs >= 3 and (new_secs < old_secs * 0.6 or
                                  len(new_content) < len(current_plan) * 0.5):
                return current_plan, (
                    "修改未生效：结果疑似缺失章节（已保留原文）。"
                    "可改用左侧选中单个章节后再修改。"
                )
            return new_content, desc
        except Exception as exc:
            return current_plan, f"修改失败：{str(exc)[:40]}"


# ── 公共工具 ──────────────────────────────────────────────────────────────────

def splice_section_into_plan(plan: str, section_keyword: str, new_section_content: str) -> str:
    """将方案中指定章节替换为新内容，返回完整方案。供手动编辑保存使用。
    定位不到目标章节时，返回原方案（不替换），避免丢失其他章节。"""
    s, e = _find_section_bounds(plan, section_keyword)
    if s == -1:
        return plan
    return _splice_section(plan, s, e, new_section_content)


# ── 主动建议 ──────────────────────────────────────────────────────────────────

def get_proactive_suggestions(current_plan: str, llm) -> str:
    """
    扫描当前方案，返回2-3条主动优化建议。
    LLM 失败时返回空字符串。
    """
    prompt = SUGGEST_PROMPT.format(plan_preview=current_plan[:1500])
    try:
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()
    except Exception:
        return ""
