import streamlit as st
import requests
import os
import time

# ── brand palette (from the RAG-READ poster) ──
# primary yellow: #FFE500, black: #1a1a1a, white: #fff

def make_source_card(chunk_num, src_name, rank_score, preview_text):
    return f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="source-tag">#{chunk_num} — {src_name}</span>
                <span class="score-pill">{rank_score:.2f}</span>
            </div>
            <div style="margin-top: 12px; font-size: 0.88rem; color: #555; line-height: 1.65;">
                {preview_text}
            </div>
        </div>
    """

def confidence_color(score):
    if score >= 7:
        return "#1a1a1a"
    elif score >= 5:
        return "#8a6d00"
    return "#cc0000"


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG-READ",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── FULL BRAND STYLESHEET ──
# yellow background, big bold type, glassmorphism settings panel, hidden sidebar
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* ── force light mode + lock all theme variables ── */
        :root,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"] {
            color-scheme: light !important;
            --primary-color: #1a1a1a !important;
            --background-color: #FFE500 !important;
            --secondary-background-color: #f0d800 !important;
            --text-color: #1a1a1a !important;
        }

        /* ── kill sidebar completely ── */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        button[data-testid="stSidebarCollapsedControl"],
        button[data-testid="baseButton-headerNoPadding"] {
            display: none !important;
        }

        /* ── yellow everything ── */
        .stApp, .main, .block-container,
        [data-testid="stAppViewContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stMainBlockContainer"] {
            background-color: #FFE500 !important;
            color: #1a1a1a !important;
        }

        /* ── force all text to black ── */
        .stApp p, .stApp span, .stApp div, .stApp label {
            color: #1a1a1a !important;
        }

        /* ── slider and toggle tracks ── */
        .stSlider [data-testid="stThumbValue"] { color: #1a1a1a !important; }
        .stSlider [role="slider"] { background: #1a1a1a !important; }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] { color: rgba(26,26,26,0.4) !important; }

        /* ── selectbox ── */
        .stSelectbox > div > div {
            background: rgba(26,26,26,0.06) !important;
            border-color: rgba(26,26,26,0.12) !important;
            color: #1a1a1a !important;
        }
        .stSelectbox svg { fill: #1a1a1a !important; }

        /* ── toggle ── */
        .stToggle span[data-testid="stToggleLabel"] { color: #1a1a1a !important; }

        /* ── typography ── */
        html, body, [class*="st-"]:not(svg):not(i):not([class*="icon"]):not(.material-symbols-rounded) {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        .material-symbols-rounded, [data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
        }

        /* ── hide streamlit chrome ── */
        #MainMenu { display: none !important; }
        footer { display: none !important; }
        header { display: none !important; }
        .stDeployButton { display: none !important; }

        /* ── nav bar ── */
        .nav-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 0 20px 0;
        }
        .nav-logo {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #1a1a1a;
            letter-spacing: 0.06em;
        }
        .nav-right {
            display: flex;
            gap: 28px;
            align-items: center;
        }
        .nav-link {
            font-size: 0.72rem;
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            text-decoration: none;
            opacity: 0.6;
            transition: opacity 0.2s;
        }
        .nav-link:hover { opacity: 1; }

        /* ── hero section ── */
        .hero {
            padding: 10vh 0 6vh 0;
            max-width: 900px;
        }
        .hero h1 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: clamp(3rem, 6vw, 5.5rem) !important;
            font-weight: 700 !important;
            color: #1a1a1a !important;
            line-height: 1.0 !important;
            letter-spacing: -0.04em !important;
            margin: 0 !important;
            padding: 0 !important;
            text-transform: uppercase;
        }
        .hero-sub {
            font-size: 0.82rem;
            color: #1a1a1a;
            opacity: 0.5;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-top: 28px;
            font-weight: 500;
        }

        /* ── pipeline strip ── */
        .pipeline-strip {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin: 32px 0 48px 0;
            align-items: center;
        }
        .p-step {
            font-size: 0.68rem;
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 6px 16px;
            border: 1.5px solid rgba(26,26,26,0.25);
            border-radius: 24px;
            transition: all 0.2s;
        }
        .p-step:hover {
            background: #1a1a1a;
            color: #FFE500;
            border-color: #1a1a1a;
        }
        .p-arrow {
            color: rgba(26,26,26,0.3);
            font-size: 0.7rem;
            margin: 0 2px;
        }

        /* ── search box ── */
        .stTextInput > div > div > input {
            background: rgba(26,26,26,0.06) !important;
            border: 2px solid rgba(26,26,26,0.12) !important;
            border-radius: 10px !important;
            color: #1a1a1a !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            padding: 16px 20px !important;
            font-family: 'Space Grotesk', sans-serif !important;
            transition: all 0.25s ease !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #1a1a1a !important;
            background: rgba(26,26,26,0.03) !important;
            box-shadow: none !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: rgba(26,26,26,0.35) !important;
            font-weight: 500 !important;
        }

        /* ── glassmorphism settings panel ── */
        .glass-panel {
            background: rgba(255, 255, 255, 0.35);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 16px;
            padding: 28px 32px;
            margin: 20px auto 40px auto;
            max-width: 700px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        }
        .glass-panel-title {
            font-size: 0.68rem;
            font-weight: 700;
            color: #1a1a1a;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 18px;
            opacity: 0.5;
        }

        /* ── inline settings bar ── */
        .settings-label {
            font-size: 0.62rem;
            font-weight: 700;
            color: rgba(26,26,26,0.35);
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        /* ── expander inside results (Hybrid vs Dense) ── */
        .stExpander {
            background: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid rgba(26,26,26,0.08) !important;
            border-radius: 12px !important;
        }
        .stExpander > details {
            border: none !important;
        }

        /* ── slider, toggle, selectbox on yellow ── */
        .stSlider label, .stSelectbox label, .stToggle label {
            color: #1a1a1a !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }
        .stSlider [data-testid="stThumbValue"] {
            color: #1a1a1a !important;
        }

        /* ── answer section headings ── */
        h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #1a1a1a !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            margin-top: 24px !important;
        }

        /* ── metric cards ── */
        .metric-card {
            background: rgba(255,255,255,0.35);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 12px;
            padding: 22px 24px;
            margin-bottom: 14px;
        }

        /* ── source chunk cards ── */
        .source-card {
            background: rgba(255,255,255,0.3);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.4);
            border-left: 4px solid #1a1a1a;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 12px;
            transition: all 0.25s ease;
        }
        .source-card:hover {
            background: rgba(255,255,255,0.5);
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.06);
        }
        .source-tag {
            font-weight: 700;
            color: #1a1a1a;
            font-size: 0.88rem;
            letter-spacing: 0.02em;
        }
        .score-pill {
            font-size: 0.72rem;
            font-weight: 600;
            color: #1a1a1a;
            background: rgba(26,26,26,0.08);
            padding: 3px 12px;
            border-radius: 20px;
        }

        /* ── citation badges ── */
        .cite {
            background: rgba(26,26,26,0.1);
            color: #1a1a1a;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.86em;
        }

        /* ── divider ── */
        hr {
            border-color: rgba(26,26,26,0.1) !important;
        }

        /* ── empty state ── */
        .empty-state {
            text-align: center;
            padding: 80px 30px;
            margin-top: 20px;
        }
        .empty-state p {
            color: rgba(26,26,26,0.4);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
        }

        /* ── buttons ── */
        .stButton > button {
            background: transparent !important;
            color: #1a1a1a !important;
            border: 2px solid #1a1a1a !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            padding: 8px 24px !important;
            transition: all 0.2s !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
        .stButton > button:hover {
            background: #1a1a1a !important;
            color: #FFE500 !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:active {
            background: #333 !important;
            color: #FFE500 !important;
        }

        /* ── file uploader ── */
        .stFileUploader {
            background: rgba(26,26,26,0.04) !important;
            border: 2px dashed rgba(26,26,26,0.15) !important;
            border-radius: 12px !important;
            padding: 8px !important;
            transition: border-color 0.2s !important;
        }
        .stFileUploader:hover {
            border-color: rgba(26,26,26,0.35) !important;
        }
        .stFileUploader label {
            color: #1a1a1a !important;
            font-weight: 600 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }
        .stFileUploader button {
            background: transparent !important;
            color: #1a1a1a !important;
            border: 1.5px solid rgba(26,26,26,0.3) !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.72rem !important;
        }
        .stFileUploader button svg,
        .stFileUploader button .material-symbols-rounded,
        .stFileUploader button [data-testid="stIconMaterial"] {
            display: none !important;
        }
        .stFileUploader button:hover {
            background: #1a1a1a !important;
            color: #FFE500 !important;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            position: relative;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] svg,
        .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
            display: none !important;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-child::before {
            content: "📑";
            font-size: 2.2rem;
            display: block;
            margin-bottom: 10px;
        }
        .stFileUploader small { color: rgba(26,26,26,0.4) !important; }

        /* ── success/info cards ── */
        .upload-result {
            background: rgba(255,255,255,0.35);
            backdrop-filter: blur(14px);
            border-left: 4px solid #1a1a1a;
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #1a1a1a;
        }
        .file-chip {
            display: inline-block;
            background: rgba(26,26,26,0.08);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #1a1a1a;
            margin: 3px 4px 3px 0;
            letter-spacing: 0.02em;
        }

        /* ── warnings ── */
        .stAlert {
            background: rgba(255,255,255,0.3) !important;
            border: 1px solid rgba(26,26,26,0.15) !important;
            border-radius: 10px !important;
            color: #1a1a1a !important;
        }

        /* ── captions ── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: rgba(26,26,26,0.5) !important;
        }

        /* ── spinner ── */
        .stSpinner > div > div {
            border-top-color: #1a1a1a !important;
        }
    </style>
""", unsafe_allow_html=True)


# ── NAV BAR ──
st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">📑 RAG-READ</div>
        <div class="nav-right">
            <span class="nav-link">SHADWAL SINGH</span>
            <span class="nav-link">V1.0</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# ── HERO SECTION ──
st.markdown("""
    <div class="hero">
        <h1>RAG-READ:<br>A PRODUCTION GRADE RAG PIPELINE WITH HYBRID SEARCH.</h1>
        <div class="hero-sub">April – May 2026 · Hybrid Retrieval · Citation Verified</div>
    </div>
""", unsafe_allow_html=True)

# pipeline strip
st.markdown("""
    <div class="pipeline-strip">
        <span class="p-step">Ingestion</span>
        <span class="p-arrow">→</span>
        <span class="p-step">Retrieval</span>
        <span class="p-arrow">→</span>
        <span class="p-step">Generation</span>
        <span class="p-arrow">→</span>
        <span class="p-step">Evaluation</span>
        <span class="p-arrow">→</span>
        <span class="p-step">API & Dashboard</span>
    </div>
""", unsafe_allow_html=True)


# ── SETTINGS (inline, minimal) ──
st.markdown('<div class="settings-label">⚙ Settings</div>', unsafe_allow_html=True)
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    num_chunks = st.slider("Chunks", 1, 10, 5)
with col_b:
    search_blend = st.select_slider(
        "Blend",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        value=0.5,
        help="Left = keyword-heavy, right = semantic-heavy"
    )
with col_c:
    rerank_on = st.toggle("Reranking", value=True)
with col_d:
    chunk_strategy = st.selectbox("Strategy", ["All", "Fixed", "Structural", "Semantic"])

st.markdown("""<div style="height: 2px; background: rgba(26,26,26,0.06); margin: 24px 0;"></div>""", unsafe_allow_html=True)

# ── API STATUS ──
api_online = False
try:
    requests.get(f"{API_URL}/", timeout=2)
    api_online = True
except Exception:
    pass

status_dot = "🟢" if api_online else "🔴"
status_text = "Connected" if api_online else "Offline — run: python src/api.py"
st.markdown(
    f'<div style="font-size: 0.7rem; font-weight: 600; color: rgba(26,26,26,0.5); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 20px;">'
    f'{status_dot} API: {status_text}</div>',
    unsafe_allow_html=True
)

# ── STEP 1: UPLOAD ──
st.markdown(
    '<div style="font-size: 0.72rem; font-weight: 700; color: #1a1a1a; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">'
    '① Upload Your Documents</div>',
    unsafe_allow_html=True
)

upload_col, docs_col = st.columns([2, 1])

with upload_col:
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "md", "html", "htm", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, Markdown, HTML, TXT"
    )

    # save uploaded files to data/raw immediately
    if uploaded_files:
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            save_path = os.path.join(raw_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.markdown(
            f'<div class="upload-result">✓ {len(uploaded_files)} file(s) saved. Click <strong>Index Now</strong> to process them.</div>',
            unsafe_allow_html=True
        )

with docs_col:
    st.markdown('<div class="settings-label">Indexed Documents</div>', unsafe_allow_html=True)
    if api_online:
        try:
            doc_list = requests.get(f"{API_URL}/v1/documents").json()
            if doc_list:
                chips_html = "".join(
                    f'<span class="file-chip">{d["filename"]} · {d["size_kb"]}KB</span>'
                    for d in doc_list
                )
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.caption("No documents indexed yet.")
        except Exception:
            st.caption("Could not load document list.")
    else:
        st.caption("Start the API to see indexed documents.")

st.markdown("""<div style="height: 2px; background: rgba(26,26,26,0.06); margin: 16px 0;"></div>""", unsafe_allow_html=True)

# ── STEP 2: INDEX ──
st.markdown(
    '<div style="font-size: 0.72rem; font-weight: 700; color: #1a1a1a; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">'
    '② Index Documents</div>',
    unsafe_allow_html=True
)

index_btn_col, index_info_col = st.columns([1, 3])

with index_btn_col:
    index_clicked = st.button("📑 Index Now", use_container_width=True)

with index_info_col:
    st.markdown(
        '<p style="font-size: 0.78rem; color: rgba(26,26,26,0.45); margin-top: 6px;">'
        'Reads all files in <code>data/raw</code>, chunks them, and builds the search index. '
        'Only needs to be done once per set of documents.</p>',
        unsafe_allow_html=True
    )

if index_clicked:
    if not api_online:
        st.error("The API server is not running. Please start it first with: `python src/api.py`")
    else:
        # get list of raw files and ingest each one
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
            st.warning("No files found in data/raw. Upload some documents first!")
        else:
            files_to_index = [f for f in os.listdir(raw_dir) if not f.startswith(".")]
            progress_bar = st.progress(0, text="Starting indexing...")
            results = []

            for i, fname in enumerate(files_to_index):
                progress_bar.progress(
                    (i) / len(files_to_index),
                    text=f"Indexing: {fname}..."
                )
                fpath = os.path.join(raw_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        resp = requests.post(
                            f"{API_URL}/v1/ingest",
                            files={"file": (fname, f)},
                            params={"strategy": "structural"},
                            timeout=120
                        )
                    if resp.status_code == 200:
                        r = resp.json()
                        results.append(f'✓ **{r["filename"]}** — {r["chunks_added"]} new chunks, {r["skipped_duplicates"]} skipped')
                    else:
                        results.append(f'✗ **{fname}** — {resp.json().get("detail", "Error")}')
                except Exception as e:
                    results.append(f'✗ **{fname}** — {str(e)[:80]}')

                # small delay to avoid rate limits on the embedding API
                time.sleep(1)

            progress_bar.progress(1.0, text="Indexing complete!")
            time.sleep(0.5)
            progress_bar.empty()

            st.markdown(
                '<div class="upload-result"><strong>Indexing Results:</strong><br>' +
                '<br>'.join(results) +
                '</div>',
                unsafe_allow_html=True
            )
            st.rerun()

st.markdown("""<div style="height: 2px; background: rgba(26,26,26,0.06); margin: 24px 0;"></div>""", unsafe_allow_html=True)

# ── STEP 3: ASK ──
st.markdown(
    '<div style="font-size: 0.72rem; font-weight: 700; color: #1a1a1a; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">'
    '③ Ask a Question</div>',
    unsafe_allow_html=True
)

user_q = st.text_input("search", placeholder="Ask anything about your documents...", label_visibility="collapsed")

if not user_q:
    st.markdown("""
        <div class="empty-state">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📑</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 8px;">How to use RAG-READ</div>
            <p style="max-width: 500px; margin: 0 auto;">
                <strong>Step 1.</strong> Upload your documents (PDF, MD, HTML, TXT) above.<br>
                <strong>Step 2.</strong> Click <strong>Index Now</strong> to process and index them.<br>
                <strong>Step 3.</strong> Type a question here to get a cited, verified answer.
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Running pipeline..."):

        strat_val = chunk_strategy.lower() if chunk_strategy != "All" else None

        req_body = {
            "query": user_q,
            "top_k": num_chunks,
            "alpha": search_blend,
            "use_reranker": rerank_on,
            "strategy": strat_val
        }

        raw = requests.post(f"{API_URL}/v1/ask", json=req_body)
        data = raw.json()

        if "detail" in data:
            st.error(data["detail"])

        else:
            left_col, right_col = st.columns([2.4, 1])

            with left_col:
                st.markdown("### Answer")

                if data["can_answer"]:
                    answer_text = data["answer"]
                    answer_html = answer_text.replace("[", '<span class="cite">[').replace("]", "]</span>")
                    st.markdown(
                        f'<div style="font-size: 1.02rem; line-height: 1.8; color: #333;">{answer_html}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Not enough info to answer confidently")
                    refusal_info = data.get("refusal", {})
                    if refusal_info:
                        st.markdown(f"**Found:** {refusal_info.get('what_is_found', '—')}")
                        st.markdown(f"**Missing:** {refusal_info.get('what_is_missing', '—')}")

            with right_col:
                st.markdown("### Scores")

                conf_score = data["confidence_score"]
                col = confidence_color(conf_score)

                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.68rem; color: rgba(26,26,26,0.45); font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">
                            Retrieval Confidence
                        </div>
                        <div style="font-size: 2.6rem; font-weight: 700; color: {col}; margin-top: 6px; font-family: 'Space Grotesk', sans-serif;">
                            {conf_score}<span style="font-size: 1rem; color: rgba(26,26,26,0.3);">/10</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                cov = data["citation_coverage"]
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.68rem; color: rgba(26,26,26,0.45); font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">
                            Citation Coverage
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a1a; margin-top: 6px; font-family: 'Space Grotesk', sans-serif;">
                            {cov}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # hybrid vs dense comparison
            with st.expander("Compare: Hybrid vs Dense-only"):
                dense_req = req_body.copy()
                dense_req["alpha"] = 1.0
                dense_req["use_reranker"] = False

                dense_data = requests.post(f"{API_URL}/v1/ask", json=dense_req).json()

                col_dense, col_hybrid = st.columns(2)

                with col_dense:
                    st.caption("DENSE ONLY")
                    for s in dense_data.get("sources", [])[:3]:
                        st.markdown(
                            f'<div style="font-size: 0.78rem; padding: 8px 12px; background: rgba(255,255,255,0.3); border: 1px solid rgba(255,255,255,0.4); border-radius: 8px; margin-bottom: 6px; color: #555;">📄 {s["source"]} &nbsp;·&nbsp; id: {s["id"][:10]}...</div>',
                            unsafe_allow_html=True
                        )

                with col_hybrid:
                    st.caption("HYBRID (BM25 + SEMANTIC)")
                    for s in data.get("sources", [])[:3]:
                        st.markdown(
                            f'<div style="font-size: 0.78rem; padding: 8px 12px; background: rgba(26,26,26,0.06); border: 1px solid rgba(26,26,26,0.1); border-radius: 8px; margin-bottom: 6px; color: #1a1a1a; font-weight: 500;">📄 {s["source"]} &nbsp;·&nbsp; {s["score"]:.4f}</div>',
                            unsafe_allow_html=True
                        )

            # source chunks
            st.markdown("### Source Chunks")

            sources = data.get("sources", [])
            for i, s in enumerate(sources):
                card_html = make_source_card(i + 1, s["source"], s["score"], s["content_preview"])
                st.markdown(card_html, unsafe_allow_html=True)
