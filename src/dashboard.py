import streamlit as st
import requests

# quick and dirty helper - just wraps the long inline html into something readable
def make_source_card(chunk_num, src_name, rank_score, preview_text):
    return f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #1e40af;">#{chunk_num} — {src_name}</span>
                <span style="font-size: 0.8rem; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 12px;">score: {rank_score:.2f}</span>
            </div>
            <div style="margin-top: 10px; font-size: 0.95rem; color: #475569; line-height: 1.5;">
                {preview_text}
            </div>
        </div>
    """

# pick the right color for the confidence number
# green if high, orange if medium, red if low
def confidence_color(score):
    if score >= 7:
        return "#10b981"
    elif score >= 5:
        return "#f59e0b"
    return "#ef4444"


API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAGRead", page_icon="📖", layout="wide")

# injecting styles manually because streamlit's native theming is too limited
# tried the config.toml approach first but couldn't get the card shadows to work that way
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        .main { background-color: #f8f9fa; }

        .metric-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
            margin-bottom: 15px;
        }

        .source-card {
            background: #fff;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: box-shadow 0.2s;
        }

        .source-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.09);
        }

        h3 { color: #1e293b !important; margin-top: 22px !important; }

        /* little blue pill for citations like [1] */
        .cite {
            background: #eff6ff;
            color: #1e40af;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# sidebar controls
with st.sidebar:
    st.markdown("### Settings")

    num_chunks = st.slider("How many chunks to retrieve", 1, 10, 5)

    # 0 = pure keyword, 1 = pure semantic, 0.5 is the balanced sweet spot
    search_blend = st.select_slider(
        "Search blend",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        value=0.5,
        help="Left = keyword-heavy, right = semantic-heavy"
    )

    rerank_on = st.toggle("LLM reranking", value=True)
    chunk_strategy = st.selectbox("Chunking strategy", ["All", "Fixed", "Structural", "Semantic"])

    st.divider()
    st.markdown("### Indexed files")

    if st.button("Refresh", use_container_width=True):
        doc_list = requests.get(f"{API_URL}/v1/documents").json()
        for d in doc_list:
            st.caption(f"• {d['filename']}  ({d['size_kb']} KB)")


# --- page ---

st.title("RAGRead")
st.markdown("#### *Hybrid Retrieval — Citation Verified*")
st.markdown("")  # a bit of breathing room

user_q = st.text_input("", placeholder="Ask anything about your documents...", label_visibility="collapsed")

if not user_q:
    st.markdown("""
        <div style="text-align:center; padding:60px 20px; background:#fff; border-radius:16px; border: 1px dashed #cbd5e1; margin-top:20px;">
            <div style="font-size:2.5rem;">📖</div>
            <h3 style="color:#1e293b; margin-top:12px;">Ask a question to get started</h3>
            <p style="color:#64748b;">The pipeline will retrieve, rerank, and verify before answering.</p>
        </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Retrieving and verifying..."):

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
            left_col, right_col = st.columns([2.2, 1])

            with left_col:
                st.markdown("### Answer")

                if data["can_answer"]:
                    # wrap citation brackets in a styled span
                    # doing it this way instead of regex because simpler and it works fine
                    answer_text = data["answer"]
                    answer_html = answer_text.replace("[", '<span class="cite">[').replace("]", "]</span>")
                    st.markdown(
                        f'<div style="font-size:1.05rem; line-height:1.7; color:#334155;">{answer_html}</div>',
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

                # rendering cards manually since st.metric doesn't let me style the number size
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#64748b; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">Retrieval Confidence</div>
                        <div style="font-size:2.4rem; font-weight:800; color:{col}; margin-top:4px;">
                            {conf_score}<span style="font-size:1rem; color:#94a3b8;">/10</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                cov = data["citation_coverage"]
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem; color:#64748b; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">Citation Coverage</div>
                        <div style="font-size:2rem; font-weight:700; color:#1e3a8a; margin-top:4px;">{cov}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # hybrid vs dense comparison
            # useful for showing why we built the hybrid retriever in the first place
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
                            f'<div style="font-size:0.8rem; padding:6px 8px; background:#f1f5f9; border-radius:4px; margin-bottom:5px;">📄 {s["source"]} &nbsp;·&nbsp; id: {s["id"][:10]}...</div>',
                            unsafe_allow_html=True
                        )

                with col_hybrid:
                    st.caption("HYBRID (BM25 + SEMANTIC)")
                    for s in data.get("sources", [])[:3]:
                        st.markdown(
                            f'<div style="font-size:0.8rem; padding:6px 8px; background:#eff6ff; border-radius:4px; margin-bottom:5px;">📄 {s["source"]} &nbsp;·&nbsp; {s["score"]:.4f}</div>',
                            unsafe_allow_html=True
                        )

            # show the retrieved chunks
            st.markdown("### Source Chunks")

            sources = data.get("sources", [])
            for i, s in enumerate(sources):
                card_html = make_source_card(i + 1, s["source"], s["score"], s["content_preview"])
                st.markdown(card_html, unsafe_allow_html=True)
