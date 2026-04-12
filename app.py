import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Cathay Shield — 智能授信助手",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
<style>
.banner{background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;
        padding:8px 14px;font-size:12px;color:#991B1B;margin-bottom:12px}
.metric-box{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
            padding:12px;text-align:center}
.metric-val{font-size:22px;font-weight:700;color:#0B2B5C}
.metric-lbl{font-size:11px;color:#64748B;margin-top:2px}
.risk-high{background:#FEF2F2;border:2px solid #DC2626;border-radius:8px;padding:12px;text-align:center}
.risk-mid{background:#FFFBEB;border:2px solid #F59E0B;border-radius:8px;padding:12px;text-align:center}
.risk-low{background:#F0FDF4;border:2px solid #16A34A;border-radius:8px;padding:12px;text-align:center}
.report-box{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
            padding:16px;font-size:13px;line-height:1.8;color:#1E293B}
.src-chip{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;
          padding:5px 10px;font-size:11px;color:#1D4ED8;margin-bottom:5px}
.step-done{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;
           padding:10px 14px;margin-bottom:6px;font-size:12px;color:#166534}
.step-running{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
              padding:10px 14px;margin-bottom:6px;font-size:12px;color:#1D4ED8}
.step-wait{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
           padding:10px 14px;margin-bottom:6px;font-size:12px;color:#94A3B8}
.agent-header{background:#0B2B5C;border-radius:10px;padding:12px 16px;
              color:#fff;font-size:13px;font-weight:600;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🏦 Cathay Shield")
st.markdown("**智能授信助手** — AI Agent Prototype")
st.caption("國泰世華 CMA 儲備幹部計畫 · AI 商業分析組 · 基於 GAIA 2.0 框架")

st.markdown("""
<div class="banner">
⚠️ <strong>Copilot 聲明：</strong>
本系統為輔助工具，AI 分析報告僅供參考，最終授信決策由授信人員確認，
符合金管會《金融業運用人工智慧核心原則》。
</div>
""", unsafe_allow_html=True)

# ── Fallback 回應（API 掛掉時使用）───────────────────────────
FALLBACK_REPORT = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
授信分析報告草稿（Fallback 模式）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
申請企業：新創科技有限公司
風險評級：高風險

【一、財務健診摘要】
本企業財務指標呈現多項警示訊號。流動比率 0.63 低於標準 1.5，短期償債能力不足；
負債比率 86.7% 遠超行內上限 70%，財務槓桿偏高；ROE 為負值，顯示目前處於虧損狀態。

【二、主要風險點】
・流動比率 0.63，低於最低標準 1.0，短期資金缺口明顯
・負債比率 86.7%，超過新創企業上限 70%，債務壓力沉重
・利息保障倍數 0.5，低於最低標準 1.5，獲利無法覆蓋利息費用

【三、授信建議】
⛔ 不建議核貸。本案財務指標未達行內標準，建議待財務結構改善後重新申請。
若確有業務往來需求，建議以充足擔保品及個人連帶保證為前提進一步評估。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 本報告由 Cathay Shield AI 輔助產出（Fallback 模式），最終授信決策請授信人員確認。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

# ── 初始化知識庫 ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ 載入授信政策知識庫...")
def init_kb():
    import glob
    from sentence_transformers import SentenceTransformer
    base = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base, "data", "*.txt"))
    all_chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            text = fp.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        all_chunks.extend(paragraphs)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks)
    return all_chunks, embeddings, model

try:
    chunks, embeddings, embed_model = init_kb()
    st.success("✅ 授信政策知識庫就緒（" + str(len(chunks)) + " 個知識片段）")
except Exception as e:
    st.error("❌ 知識庫載入失敗：" + str(e))
    st.stop()

# ── 檢索函式 ──────────────────────────────────────────────────
def retrieve(query, top_k=3):
    from sklearn.metrics.pairwise import cosine_similarity
    q_emb = embed_model.encode([query])[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]

# ── 財務指標計算 ──────────────────────────────────────────────
def calc_metrics(data):
    metrics = {}
    if data["current_assets"] > 0 and data["current_liabilities"] > 0:
        metrics["流動比率"] = round(data["current_assets"] / data["current_liabilities"], 2)
    if data["total_assets"] > 0:
        metrics["負債比率"] = round(data["total_liabilities"] / data["total_assets"] * 100, 1)
    if data["interest_expense"] > 0:
        metrics["利息保障倍數"] = round(data["ebit"] / data["interest_expense"], 2)
    if data["equity"] > 0:
        metrics["ROE"] = round(data["net_income"] / data["equity"] * 100, 1)

    score = 0
    if metrics.get("流動比率", 0) >= 1.5: score += 1
    if metrics.get("負債比率", 100) <= 60: score += 1
    if metrics.get("利息保障倍數", 0) >= 3.0: score += 1
    if metrics.get("ROE", 0) >= 8: score += 1

    risk = "低風險" if score >= 3 else ("中風險" if score == 2 else "高風險")
    return metrics, risk

# ── GenAI 生成報告（含 fallback）────────────────────────────
def generate_report(company, metrics, risk, industry, loan_amount, policy_chunks):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return FALLBACK_REPORT, True  # (報告內容, 是否為fallback)

    policy_text = "\n\n".join(policy_chunks)
    metrics_text = "\n".join([f"- {k}：{v}" for k, v in metrics.items()])

    prompt = (
        "你是國泰世華銀行資深授信分析師。請根據財務指標和授信政策，"
        "產出一份專業的授信分析報告草稿。\n\n"
        "【企業資訊】\n"
        "公司名稱：" + company + "\n"
        "產業別：" + industry + "\n"
        "申請額度：NT$" + str(loan_amount) + " 萬元\n"
        "AI 風險評級：" + risk + "\n\n"
        "【財務指標】\n" + metrics_text + "\n\n"
        "【相關授信政策】\n" + policy_text + "\n\n"
        "請用繁體中文輸出以下格式：\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "授信分析報告草稿\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "申請企業：" + company + "\n"
        "風險評級：" + risk + "\n\n"
        "【一、財務健診摘要】\n（2–3句說明財務狀況整體評估）\n\n"
        "【二、主要優勢】\n（條列 2–3 項財務強項，若為高風險則說明主要風險點）\n\n"
        "【三、主要風險點】\n（條列 2–3 項需關注事項）\n\n"
        "【四、授信建議】\n（建議核貸/謹慎核貸/不建議核貸，及條件說明，引用授信政策條文）\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ 本報告由 Cathay Shield AI 輔助產出，最終授信決策請授信人員確認。\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        return response.text, False
    except Exception:
        return FALLBACK_REPORT, True

# ── 步驟動畫函式 ──────────────────────────────────────────────
def show_agent_steps(placeholder, current_step, steps_status):
    """
    steps_status: list of "wait" | "running" | "done"
    """
    steps = [
        ("🔧 Tool 1", "規則引擎", "財務指標自動計算（本地執行，不耗 API）"),
        ("🔍 Tool 2", "RAG 檢索", "查詢授信手冊知識庫，召回相關政策條文"),
        ("🧠 Skill 1", "LLM 推理", "異常解讀 + 5P 分析 + 風險評級（Gemini）"),
        ("📄 Skill 2", "結構化輸出", "依行內模板產出授信分析報告草稿"),
    ]
    with placeholder.container():
        st.markdown('<div class="agent-header">🤖 AI Agent 執行中 — GAIA 2.0 框架</div>', unsafe_allow_html=True)
        for i, (tag, title, desc) in enumerate(steps):
            status = steps_status[i]
            if status == "done":
                st.markdown(
                    f'<div class="step-done">✅ <strong>{tag} · {title}</strong> — {desc}</div>',
                    unsafe_allow_html=True
                )
            elif status == "running":
                st.markdown(
                    f'<div class="step-running">⏳ <strong>{tag} · {title}</strong> — {desc} ...</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="step-wait">○ <strong>{tag} · {title}</strong> — {desc}</div>',
                    unsafe_allow_html=True
                )

# ── 主介面 ────────────────────────────────────────────────────
st.divider()
st.subheader("📋 輸入企業財務資料")

CASES = {
    "🟢 健康案例：台灣製造股份有限公司": {
        "industry": "製造業", "loan_amount": 500,
        "current_assets": 8000, "current_liabilities": 4000,
        "total_assets": 20000, "total_liabilities": 10000,
        "ebit": 3000, "interest_expense": 500,
        "net_income": 2000, "equity": 10000,
    },
    "🔴 高風險案例：新創科技有限公司": {
        "industry": "科技新創", "loan_amount": 300,
        "current_assets": 500, "current_liabilities": 800,
        "total_assets": 3000, "total_liabilities": 2600,
        "ebit": 100, "interest_expense": 200,
        "net_income": -150, "equity": 400,
    },
    "🟡 中等風險：成長服務股份有限公司": {
        "industry": "服務業", "loan_amount": 200,
        "current_assets": 2400, "current_liabilities": 1800,
        "total_assets": 8000, "total_liabilities": 4400,
        "ebit": 800, "interest_expense": 400,
        "net_income": 300, "equity": 3600,
    },
}

selected = st.selectbox("選擇預設案例", list(CASES.keys()))
case = CASES[selected]

col1, col2 = st.columns(2)
with col1:
    company = st.text_input("公司名稱", value=selected.split("：")[1])
    industry = st.text_input("產業別", value=case["industry"])
    loan_amount = st.number_input("申請額度（萬元）", value=case["loan_amount"], min_value=1)
    current_assets = st.number_input("流動資產（萬元）", value=case["current_assets"])
    current_liabilities = st.number_input("流動負債（萬元）", value=case["current_liabilities"])
    total_assets = st.number_input("總資產（萬元）", value=case["total_assets"])
with col2:
    total_liabilities = st.number_input("總負債（萬元）", value=case["total_liabilities"])
    ebit = st.number_input("EBIT 稅前息前獲利（萬元）", value=case["ebit"])
    interest_expense = st.number_input("利息費用（萬元）", value=case["interest_expense"])
    net_income = st.number_input("稅後淨利（萬元）", value=case["net_income"])
    equity = st.number_input("股東權益（萬元）", value=case["equity"])

if st.button("🚀 啟動 AI Agent 分析", type="primary", use_container_width=True):

    data = {
        "current_assets": current_assets, "current_liabilities": current_liabilities,
        "total_assets": total_assets, "total_liabilities": total_liabilities,
        "ebit": ebit, "interest_expense": interest_expense,
        "net_income": net_income, "equity": equity,
    }

    st.divider()

    # ── Agent 步驟動畫區 ──────────────────────────────────────
    agent_placeholder = st.empty()

    # Step 1：規則引擎
    show_agent_steps(agent_placeholder, 0, ["running", "wait", "wait", "wait"])
    metrics, risk = calc_metrics(data)
    time.sleep(0.5)
    show_agent_steps(agent_placeholder, 1, ["done", "wait", "wait", "wait"])

    # Step 2：RAG 檢索
    show_agent_steps(agent_placeholder, 1, ["done", "running", "wait", "wait"])
    query = f"{industry} 授信 財務比率 {risk}"
    policy_chunks = retrieve(query)
    time.sleep(0.8)
    show_agent_steps(agent_placeholder, 2, ["done", "done", "wait", "wait"])

    # Step 3：LLM 推理
    show_agent_steps(agent_placeholder, 2, ["done", "done", "running", "wait"])
    report, is_fallback = generate_report(company, metrics, risk, industry, loan_amount, policy_chunks)
    show_agent_steps(agent_placeholder, 3, ["done", "done", "done", "wait"])

    # Step 4：結構化輸出
    show_agent_steps(agent_placeholder, 3, ["done", "done", "done", "running"])
    time.sleep(0.3)
    show_agent_steps(agent_placeholder, 4, ["done", "done", "done", "done"])

    # ── 財務指標結果 ───────────────────────────────────────────
    st.subheader("📊 財務指標分析結果")

    risk_class = {"高風險": "risk-high", "中風險": "risk-mid", "低風險": "risk-low"}[risk]
    risk_color = {"高風險": "#DC2626", "中風險": "#F59E0B", "低風險": "#16A34A"}[risk]
    st.markdown(
        f'<div class="{risk_class}"><div style="font-size:20px;font-weight:700;color:{risk_color}">'
        f'AI 風險評級：{risk}</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    standards = {
        "流動比率": ("≥ 1.5", 1.5, False),
        "負債比率": ("≤ 60%", 60, True),
        "利息保障倍數": ("≥ 3.0", 3.0, False),
        "ROE": ("≥ 8%", 8, False),
    }
    cols = st.columns(4)
    for i, (k, v) in enumerate(metrics.items()):
        std_label, std_val, reverse = standards.get(k, ("", 0, False))
        ok = (v <= std_val) if reverse else (v >= std_val)
        color = "#16A34A" if ok else "#DC2626"
        status = "✅" if ok else "⚠️"
        with cols[i]:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                f'{status} {v}{"%" if "比率" in k or "ROE" in k else "x"}</div>'
                f'<div class="metric-lbl">{k}</div>'
                f'<div style="font-size:10px;color:#94A3B8">標準：{std_label}</div></div>',
                unsafe_allow_html=True
            )

    # ── 報告輸出 ───────────────────────────────────────────────
    st.divider()
    st.subheader("📄 授信分析報告草稿")

    if is_fallback:
        st.warning("⚠️ API 連線失敗，已自動切換 Fallback 模式（展示用預存報告）")

    st.markdown(
        '<div class="report-box">' + report.replace("\n", "<br>") + '</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ 下載授信分析報告",
        data=report,
        file_name=f"credit_report_{company}.txt",
        mime="text/plain",
        use_container_width=True
    )

    with st.expander("📚 引用授信政策條文"):
        for i, src in enumerate(policy_chunks, 1):
            st.markdown(
                f'<div class="src-chip">📎 條文 {i}：{src[:120]}...</div>',
                unsafe_allow_html=True
            )

# ── 側邊欄 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Cathay Shield")
    st.markdown("**AI Agent 架構**")
    st.markdown("""
**Tools（外部功能呼叫）**
- 🔧 Tool 1：規則引擎（財務計算）
- 🔍 Tool 2：RAG 檢索（知識庫）

**Skills（LLM 內建能力）**
- 🧠 Skill 1：推理分析
- 📄 Skill 2：結構化輸出
""")
    st.divider()
    st.code("LLM: Gemini 2.5 Flash\nEmbedding: 本地模型\n計算: 規則引擎\n框架: LangGraph（概念）\n前端: Streamlit", language="text")
    st.divider()
    st.caption("基於 GAIA 2.0 框架設計\nCopilot 定位 · 決策歸授信人員")