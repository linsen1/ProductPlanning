"""
博查 AI 搜索连通性 + 解析自检脚本（独立运行，不依赖 streamlit）
用法：
    cd ProjectAI
    python3 test_bocha_aisearch.py
    # 或指定查询： python3 test_bocha_aisearch.py "健康管理师 培训机构 竞品对比"
"""
import sys, json, requests
import config   # 复用 .env 里的 BOCHA_API_KEY / URL / 参数

query = sys.argv[1] if len(sys.argv) > 1 else "健康管理师 培训机构 竞品对比：主流机构、卖点、价格、口碑、弱点"

print("=" * 60)
print("BOCHA key:", (config.BOCHA_API_KEY[:6] + "..." + config.BOCHA_API_KEY[-4:]) if config.BOCHA_API_KEY else "未配置")
print("URL:", config.BOCHA_AISEARCH_URL)
print("query:", query)
print("=" * 60)

try:
    resp = requests.post(
        config.BOCHA_AISEARCH_URL,
        headers={"Authorization": f"Bearer {config.BOCHA_API_KEY}", "Content-Type": "application/json"},
        json={"query": query, "freshness": config.BOCHA_FRESHNESS,
              "answer": True, "stream": False, "count": config.BOCHA_SEARCH_COUNT},
        timeout=40,
    )
    print("HTTP 状态:", resp.status_code)
    resp.raise_for_status()
    d = resp.json()
except Exception as e:
    print("❌ 调用失败:", type(e).__name__, str(e)[:300])
    sys.exit(1)

print("顶层字段:", list(d.keys()), "| code:", d.get("code"), "| msg:", d.get("msg") or d.get("message"))

# ↓↓↓ 与 app.py bocha_ai_search 完全一致的解析逻辑 ↓↓↓
msgs = (d.get("data", {}) or {}).get("messages") or d.get("messages") or []
print("messages 条数:", len(msgs), "| 各条 type:", [m.get("type") for m in msgs])

answer, pages = "", []
for m in msgs:
    mtype = m.get("type", "")
    content = m.get("content", "")
    if mtype == "answer":
        answer += content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    elif mtype == "source":
        obj = content
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                obj = {}
        vals = obj.get("value") or (obj.get("webPages", {}) or {}).get("value") or []
        pages += [v for v in vals if isinstance(v, dict)]

print("=" * 60)
print(f"✅ 解析结果：综合答案 {len(answer)} 字 | 来源网页 {len(pages)} 条")
print("-" * 60)
print("【综合答案预览】\n", (answer[:600] + ("..." if len(answer) > 600 else "")) or "（空——可能该接口不返回 answer，需检查 answer=True 是否生效）")
print("-" * 60)
print("【来源样例】")
for p in pages[:3]:
    print(" -", p.get("name"), "|", p.get("url"))
if not answer and not pages:
    print("\n⚠️ 答案和来源都为空 —— 请把上面打印的【顶层字段/messages type】发我，我据真实结构调解析。")
