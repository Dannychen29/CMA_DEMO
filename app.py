"""
app.py — Cathay Shield UI 層
只負責 Streamlit 介面，所有 AI / RAG 邏輯在獨立服務模組：
  - rag_service.py     向量知識庫
  - gemini_service.py  LLM 報告生成
"""

import streamlit as st
import time
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── 服務層匯入 ────────────────────────────────────────────────────────────────
from rag_service import init_kb, retrieve, build_rag_query
from gemini_service import generate_report, get_review_level, FALLBACK_REPORT

# ── PDF 解析 ──────────────────────────────────────────────────────────────────
FIELD_LABELS = {
    "current_assets":      "流動資產",
    "current_liabilities": "流動負債",
    "total_assets":        "總資產",
    "total_liabilities":   "總負債",
    "ebit":                "EBIT（營業利益）",
    "interest_expense":    "利息費用",
    "net_income":          "稅後淨利",
    "equity":              "股東權益",
}

def parse_pdf(uploaded_file):
    """
    解析財報 PDF。
    回傳 (result_dict | None, message_str)

    result_dict 只在 8 個欄位全部成功時才回傳完整 dict；
    部分成功時回傳 None + 說明哪些欄位缺失，讓使用者手動填入，
    避免 PDF 數字和預設值混搭導致財務指標計算錯誤。
    """
    try:
        import pdfplumber
    except ImportError:
        return None, "請先執行 pip install pdfplumber"

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join(
                (page.extract_text() or "") + "\n" for page in pdf.pages
            )
    except Exception as e:
        return None, f"PDF 開啟失敗：{e}"

    if not text.strip():
        return None, (
            "⚠️ 此 PDF 為掃描圖片格式，無法直接抽取文字。"
            "請確認財報為可選取文字的 PDF，或手動填入下方數據。"
        )

    text = text.replace(",", "").replace("，", "")
    patterns = {
        "current_assets":      r"流動資產[^\d-]*?([\d]+(?:\.\d+)?)",
        "current_liabilities": r"流動負債[^\d-]*?([\d]+(?:\.\d+)?)",
        "total_assets":        r"資產總[計額][^\d-]*?([\d]+(?:\.\d+)?)",
        "total_liabilities":   r"負債總[計額][^\d-]*?([\d]+(?:\.\d+)?)",
        "ebit":                r"營業利[益潤][^\d-]*?([\d]+(?:\.\d+)?)",
        "interest_expense":    r"利息費用[^\d-]*?([\d]+(?:\.\d+)?)",
        "net_income":          r"本期(?:稅後)?(?:淨利|損益)[^\d-]*?(-?[\d]+(?:\.\d+)?)",
        "equity":              r"股東權益[^\d-]*?([\d]+(?:\.\d+)?)",
    }

    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        result[key] = float(m.group(1)) if m else None

    found   = [k for k, v in result.items() if v is not None]
    missing = [k for k, v in result.items() if v is None]

    if len(found) == 0:
        return None, "⚠️ 找到文字但無法比對任何財務欄位，請手動填入數據。"

    if missing:
        # 部分解析 → 不採用，告知使用者缺哪些欄位
        missing_labels = "、".join(FIELD_LABELS[k] for k in missing)
        found_labels   = "、".join(FIELD_LABELS[k] for k in found)
        msg = (
            f"⚠️ 僅成功解析 {len(found)}/8 個欄位（{found_labels}），"
            f"缺少：{missing_labels}。\n"
            "為避免混搭數字導致計算錯誤，已保留下方預設值，請手動修正。"
        )
        return None, msg

    # 全部 8 個欄位都成功 → 回傳完整結果
    return result, f"✅ 成功解析全部 8/8 個財務欄位，已自動填入。"


# ── 財務計算 ──────────────────────────────────────────────────────────────────
def calc_metrics(data: dict) -> tuple[dict, str, int]:
    metrics = {}
    if data.get("current_assets", 0) > 0 and data.get("current_liabilities", 0) > 0:
        metrics["流動比率"] = round(data["current_assets"] / data["current_liabilities"], 2)
    if data.get("total_assets", 0) > 0:
        metrics["負債比率"] = round(data["total_liabilities"] / data["total_assets"] * 100, 1)
    if data.get("interest_expense", 0) > 0:
        metrics["利息保障倍數"] = round(data["ebit"] / data["interest_expense"], 2)
    if data.get("equity", 0) > 0:
        metrics["ROE"] = round(data["net_income"] / data["equity"] * 100, 1)

    score = sum([
        metrics.get("流動比率", 0) >= 1.5,
        metrics.get("負債比率", 100) <= 60,
        metrics.get("利息保障倍數", 0) >= 3.0,
        metrics.get("ROE", 0) >= 8,
    ])
    risk = "低風險" if score >= 3 else ("中風險" if score == 2 else "高風險")
    return metrics, risk, score


# ── Step 進度條 ───────────────────────────────────────────────────────────────
def render_steps(statuses: list, placeholder):
    steps_info = [
        ("🔧", "Tool 1 · 規則引擎",    "財務指標計算，不耗 API"),
        ("🔍", "Tool 2 · RAG 檢索",    "語意比對授信手冊條文（向量快取）"),
        ("🧠", "Skill 1 · LLM 推理",   "五P分析 + 異常解讀 + 風險評級"),
        ("📄", "Skill 2 · 結構化輸出", "依行內模板產出報告草稿"),
    ]
    css_map  = {"wait": "step-wait", "running": "step-run", "done": "step-done"}
    icon_map = {"wait": "○", "running": "⏳", "done": "✅"}
    html = ""
    for i, (em, title, sub) in enumerate(steps_info):
        s = statuses[i]
        html += (
            f'<div class="step-box {css_map[s]}">'
            f'<span style="font-size:14px;flex-shrink:0">{icon_map[s]}</span>'
            f'<div><div style="font-weight:600">{em} {title}</div>'
            f'<div style="font-size:10px;opacity:.75;margin-top:1px">{sub}</div></div></div>'
        )
    placeholder.markdown(html, unsafe_allow_html=True)


# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cathay Shield", page_icon="🏦", layout="wide")
st.markdown("""
<style>
.banner{background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;
        padding:7px 13px;font-size:11px;color:#991B1B;margin-bottom:10px}
.step-box{border-radius:8px;padding:9px 12px;margin-bottom:5px;font-size:11px;
          display:flex;align-items:flex-start;gap:9px}
.step-wait{background:#F8FAFC;border:1px solid #E2E8F0;color:#94A3B8}
.step-run {background:#EFF6FF;border:1px solid #BFDBFE;color:#1D4ED8}
.step-done{background:#F0FDF4;border:1px solid #BBF7D0;color:#166534}
.metric-card{background:#fff;border:1px solid #E2E8F0;border-radius:8px;
             padding:9px;text-align:center}
.mval{font-size:18px;font-weight:700;margin-bottom:2px}
.mlbl{font-size:10px;color:#64748B}
.mstd{font-size:9px;color:#94A3B8;margin-top:1px}
.risk-high{background:#FEF2F2;border:1.5px solid #DC2626;border-radius:7px;
           padding:7px;text-align:center;margin-bottom:7px}
.risk-mid {background:#FFFBEB;border:1.5px solid #F59E0B;border-radius:7px;
           padding:7px;text-align:center;margin-bottom:7px}
.risk-low {background:#F0FDF4;border:1.5px solid #16A34A;border-radius:7px;
           padding:7px;text-align:center;margin-bottom:7px}
.report-box{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
            padding:12px;font-size:11px;line-height:1.75;color:#1E293B;
            height:390px;overflow-y:auto}
.src-chip{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:5px;
          padding:3px 8px;font-size:10px;color:#1D4ED8;margin-bottom:3px;display:block}
.col-hdr{font-size:11px;font-weight:600;color:#0B2B5C;text-transform:uppercase;
         letter-spacing:.5px;margin-bottom:8px;padding-bottom:5px;
         border-bottom:2px solid #0B2B5C}
.upload-hint{background:#F8FAFC;border:1.5px dashed #CBD5E0;border-radius:8px;
             padding:10px;text-align:center;font-size:11px;color:#94A3B8;margin-bottom:7px}
</style>
""", unsafe_allow_html=True)

# ── 知識庫初始化（快取加速）────────────────────────────────────────────────────
@st.cache_resource(show_spinner="載入授信政策知識庫...")
def load_kb():
    return init_kb()

try:
    _chunks, _embeddings, _model = load_kb()
except Exception as e:
    st.error(f"知識庫載入失敗：{e}")
    st.stop()

# ── 頁首 ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
    "<span style='font-size:24px'>🏦</span>"
    "<div><strong style='font-size:16px'>Cathay Shield</strong> "
    "<span style='color:#64748B;font-size:13px'>"
    "智能授信助手 — AI Agent Prototype</span></div></div>",
    unsafe_allow_html=True,
)
st.caption("國泰世華 CMA 儲備幹部計畫 · AI 商業分析組 · 基於 GAIA 2.0 框架")
st.markdown(
    '<div class="banner">⚠️ <strong>Copilot 聲明：</strong>'
    '本系統為輔助工具，AI 分析報告僅供參考，最終授信決策由授信人員確認，'
    '符合金管會《金融業運用人工智慧核心原則》。</div>',
    unsafe_allow_html=True,
)

# ── 預設案例 ──────────────────────────────────────────────────────────────────
CASES = {
    "🔴 高風險：新創科技有限公司": {
        "industry": "科技新創", "loan_amount": 300,
        "current_assets": 500,  "current_liabilities": 800,
        "total_assets": 3000,   "total_liabilities": 2600,
        "ebit": 100, "interest_expense": 200, "net_income": -150, "equity": 400,
    },
    "🟢 健康：台灣製造股份有限公司": {
        "industry": "製造業", "loan_amount": 500,
        "current_assets": 8000,  "current_liabilities": 4000,
        "total_assets": 20000,   "total_liabilities": 10000,
        "ebit": 3000, "interest_expense": 500, "net_income": 2000, "equity": 10000,
    },
    "🟡 中等：成長服務股份有限公司": {
        "industry": "服務業", "loan_amount": 200,
        "current_assets": 2400,  "current_liabilities": 1800,
        "total_assets": 8000,    "total_liabilities": 4400,
        "ebit": 800, "interest_expense": 400, "net_income": 300, "equity": 3600,
    },
}

# ── 三欄佈局 ──────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 1, 1.6])

with col_left:
    st.markdown('<div class="col-hdr">① 上傳財報 / 輸入資料</div>', unsafe_allow_html=True)
    selected = st.selectbox("選擇案例", list(CASES.keys()), label_visibility="collapsed")
    case = CASES[selected]

    uploaded = st.file_uploader("上傳財報 PDF", type=["pdf"], label_visibility="collapsed")
    # pdf_vals 只在 8/8 全部解析成功時才有值，避免混搭數字
    pdf_vals: dict = {}
    if uploaded:
        result, msg = parse_pdf(uploaded)
        if result is not None:
            # 8/8 全部成功 → 採用
            pdf_vals = result
            st.success(f"📄 {uploaded.name} — {msg}")
            st.caption("💡 下方數據已自動填入，可手動修正後再執行分析。")
        else:
            # 部分或完全失敗 → 保留預設值，顯示詳細缺欄位說明
            st.warning(f"📄 {uploaded.name} — PDF 解析未完整")
            st.caption(msg)
    else:
        st.markdown(
            '<div class="upload-hint">📄 上傳財報 PDF 自動解析<br>'
            '<span style="font-size:9px">8 個欄位全部成功才會自動填入，部分失敗請手動修正</span></div>',
            unsafe_allow_html=True,
        )

    def pval(key):
        """PDF 全部 8 欄成功時用 PDF 值，否則用預設案例值，絕不混搭"""
        return int(pdf_vals[key]) if key in pdf_vals else case[key]

    company     = st.text_input("公司名稱", value=selected.split("：")[1])
    c1, c2      = st.columns(2)
    with c1:
        industry    = st.text_input("產業", value=case["industry"])
    with c2:
        loan_amount = st.number_input("額度（萬）", value=case["loan_amount"], min_value=1)

    pdf_label = "財務數據（✅ 已從 PDF 填入，可手動修改）" if pdf_vals else "財務數據（使用預設案例，可手動修改）"
    with st.expander(pdf_label, expanded=bool(pdf_vals)):
        current_assets      = st.number_input("流動資產",  value=pval("current_assets"))
        current_liabilities = st.number_input("流動負債",  value=pval("current_liabilities"))
        total_assets        = st.number_input("總資產",    value=pval("total_assets"))
        total_liabilities   = st.number_input("總負債",    value=pval("total_liabilities"))
        ebit                = st.number_input("EBIT",      value=pval("ebit"))
        interest_expense    = st.number_input("利息費用",  value=pval("interest_expense"))
        net_income          = st.number_input("稅後淨利",  value=pval("net_income"))
        equity              = st.number_input("股東權益",  value=pval("equity"))

    run = st.button("🚀 啟動 AI Agent 分析", type="primary", use_container_width=True)

with col_mid:
    st.markdown('<div class="col-hdr">② Agent 執行流程</div>', unsafe_allow_html=True)
    steps_ph   = st.empty()
    metrics_ph = st.empty()

with col_right:
    st.markdown('<div class="col-hdr">③ 授信分析報告草稿</div>', unsafe_allow_html=True)
    report_ph = st.empty()
    dl_ph     = st.empty()
    src_ph    = st.empty()

render_steps(["wait", "wait", "wait", "wait"], steps_ph)
report_ph.markdown(
    '<div class="report-box" style="color:#94A3B8;display:flex;'
    'align-items:center;justify-content:center">啟動 Agent 後顯示報告草稿</div>',
    unsafe_allow_html=True,
)

# ── 分析主流程 ────────────────────────────────────────────────────────────────
if run:
    data = {
        "current_assets": current_assets, "current_liabilities": current_liabilities,
        "total_assets": total_assets,     "total_liabilities": total_liabilities,
        "ebit": ebit, "interest_expense": interest_expense,
        "net_income": net_income, "equity": equity,
    }
    standards = {
        "流動比率":     ("≥ 1.5", 1.5, False),
        "負債比率":     ("≤ 60%", 60,  True),
        "利息保障倍數": ("≥ 3.0", 3.0, False),
        "ROE":          ("≥ 8%",  8,   False),
    }

    # Step 1 — 規則引擎
    render_steps(["running", "wait", "wait", "wait"], steps_ph)
    metrics, risk, score = calc_metrics(data)
    time.sleep(0.4)
    render_steps(["done", "wait", "wait", "wait"], steps_ph)

    risk_color   = {"高風險": "#DC2626", "中風險": "#F59E0B", "低風險": "#16A34A"}[risk]
    risk_class   = {"高風險": "risk-high", "中風險": "risk-mid", "低風險": "risk-low"}[risk]
    review_level = get_review_level(loan_amount)

    m_html = (
        f'<div class="{risk_class}">'
        f'<span style="font-size:13px;font-weight:700;color:{risk_color}">風險評級：{risk}</span>'
        f'<span style="font-size:10px;color:#64748B;margin-left:8px">覆核：{review_level}</span>'
        f'</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">'
    )
    for k, v in metrics.items():
        std_label, std_val, reverse = standards.get(k, ("", 0, False))
        ok     = (v <= std_val) if reverse else (v >= std_val)
        color  = "#16A34A" if ok else "#DC2626"
        icon   = "✅" if ok else "⚠️"
        suffix = "%" if ("比率" in k or "ROE" in k) else "×"
        m_html += (
            f'<div class="metric-card">'
            f'<div class="mval" style="color:{color}">{icon} {v}{suffix}</div>'
            f'<div class="mlbl">{k}</div>'
            f'<div class="mstd">{std_label}</div></div>'
        )
    m_html += '</div>'
    metrics_ph.markdown(m_html, unsafe_allow_html=True)

    # Step 2 — RAG 檢索
    render_steps(["done", "running", "wait", "wait"], steps_ph)
    rag_query     = build_rag_query(metrics, industry)
    policy_chunks = retrieve(rag_query, top_k=4)
    time.sleep(0.5)
    render_steps(["done", "done", "wait", "wait"], steps_ph)

    # Step 3 — LLM 推理
    render_steps(["done", "done", "running", "wait"], steps_ph)
    try:
        report, is_fallback = generate_report(
            company, metrics, risk, score, industry, loan_amount, policy_chunks, data
        )
    except RuntimeError as e:
        st.error(str(e))
        report, is_fallback = FALLBACK_REPORT, True
    render_steps(["done", "done", "done", "wait"], steps_ph)

    # Step 4 — 結構化輸出
    render_steps(["done", "done", "done", "running"], steps_ph)
    time.sleep(0.3)
    render_steps(["done", "done", "done", "done"], steps_ph)

    report_ph.markdown(
        '<div class="report-box">' + report.replace("\n", "<br>") + '</div>',
        unsafe_allow_html=True,
    )
    dl_ph.download_button(
        "⬇️ 下載報告草稿",
        data=report,
        file_name=f"credit_report_{company}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    src_ph.markdown(
        '<div style="margin-top:6px">'
        '<div style="font-size:10px;font-weight:600;color:#64748B;margin-bottom:4px">'
        '📚 召回授信政策條文（Top 4）</div>'
        + "".join(
            f'<div class="src-chip">📎 {s[:100]}...</div>' for s in policy_chunks
        )
        + '</div>',
        unsafe_allow_html=True,
    )
