import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- Config ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAGRead Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --- Custom Styling & Premium Branding ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main Background & Title */
        .main {
            background-color: #f8f9fa;
        }
        
        .stTitle {
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(90deg, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 20px;
        }

        /* Glassmorphism Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }

        .source-card {
            background: #ffffff;
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }

        .source-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }

        /* Typography Fixes */
        h3 {
            font-weight: 700 !important;
            color: #1e293b !important;
            margin-top: 25px !important;
        }

        .citation-tag {
            background-color: #eff6ff;
            color: #1e40af;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛠️ Configuration")
    top_k = st.slider("Retrieval Depth (K)", 1, 10, 5)
    alpha = st.select_slider("Search Mode", 
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        value=0.5,
        help="0.0 = Pure Keyword, 0.5 = Hybrid, 1.0 = Pure Semantic"
    )
    use_reranker = st.toggle("Deep LLM Reranking", value=True)
    strategy = st.selectbox("Document Strategy", ["All", "Fixed", "Structural", "Semantic"])

    st.markdown("---")
    st.markdown("### 📂 Active Corpus")
    if st.button("Refresh Index", use_container_width=True):
        try:
            docs = requests.get(f"{API_BASE_URL}/v1/documents").json()
            for doc in docs:
                st.caption(f"• {doc['filename']} ({doc['size_kb']} KB)")
        except:
            st.error("API Offline")

# --- Main Dashboard ---
st.title("RAGRead Platform")
st.markdown("#### *Advanced Hybrid Retrieval & Factual Verification*")

query = st.text_input("", placeholder="Ask a technical or professional question...", label_visibility="collapsed")

if query:
    with st.spinner("Analyzing corpus and verifying citations..."):
        try:
            # 1. Main Request
            payload = {
                "query": query, "top_k": top_k, "alpha": alpha, 
                "use_reranker": use_reranker, 
                "strategy": strategy.lower() if strategy != "All" else None
            }
            response = requests.post(f"{API_BASE_URL}/v1/ask", json=payload).json()
            
            if "detail" in response:
                st.error(f"API Error: {response['detail']}")
            else:
                # Layout: Answer (Left) | Metrics (Right)
                col1, col2 = st.columns([2.2, 1])
                
                with col1:
                    st.markdown("### 🤖 Verified Answer")
                    if response["can_answer"]:
                        # Prettify citations
                        answer = response["answer"].replace("[", '<span class="citation-tag">[').replace("]", "]</span>")
                        st.markdown(f'<div style="font-size: 1.1rem; line-height: 1.6; color: #334155;">{answer}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Structured Refusal: Insufficient Context")
                        refusal = response["refusal"]
                        st.markdown(f"**I found info about:** {refusal['what_is_found']}")
                        st.markdown(f"**But I'm missing:** {refusal['what_is_missing']}")
                        st.info(f"💡 *{refusal['suggested_documents']}*")

                with col2:
                    st.markdown("### 📈 Quality Score")
                    
                    # Confidence Gauge
                    conf = response["confidence_score"]
                    color = "#10b981" if conf >= 7 else "#f59e0b" if conf >= 5 else "#ef4444"
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Confidence Score</div>
                            <div style="font-size: 2.5rem; font-weight: 800; color: {color};">{conf}<span style="font-size: 1rem; color: #94a3b8;">/10</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    coverage = response["citation_coverage"]
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Citation Coverage</div>
                            <div style="font-size: 2rem; font-weight: 700; color: #1e3a8a;">{coverage}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                
                # Side-by-Side Comparison (Money Shot)
                with st.expander("🔍 System Comparison: Hybrid vs. Dense Search Efficiency"):
                    col_a, col_b = st.columns(2)
                    dense_payload = payload.copy()
                    dense_payload["alpha"] = 1.0 # Pure semantic
                    dense_payload["use_reranker"] = False 
                    dense_res = requests.post(f"{API_BASE_URL}/v1/ask", json=dense_payload).json()

                    with col_a:
                        st.caption("🌌 DENSE-ONLY (SEMANTIC)")
                        for s in dense_res["sources"][:3]:
                            st.markdown(f'<div style="font-size: 0.8rem; padding: 5px; background: #f1f5f9; border-radius: 4px; margin-bottom: 5px;">📄 {s["source"]} | ID: {s["id"][:8]}...</div>', unsafe_allow_html=True)
                    with col_b:
                        st.caption("🧬 HYBRID FUSION (OUR ENGINE)")
                        for s in response["sources"][:3]:
                            st.markdown(f'<div style="font-size: 0.8rem; padding: 5px; background: #eff6ff; border-radius: 4px; margin-bottom: 5px;">📄 {s["source"]} | Score: {s["score"]:.4f}</div>', unsafe_allow_html=True)

                # Ranked Chunks
                st.markdown("### 📚 Ground Truth: Source Evidence")
                for i, source in enumerate(response["sources"]):
                    st.markdown(f"""
                        <div class="source-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; color: #1e40af;">CHUNK {i+1} — {source['source']}</span>
                                <span style="font-size: 0.8rem; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 12px;">Rank Score: {source['score']:.2f}</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.95rem; color: #475569; line-height: 1.5;">
                                {source['content_preview']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Engine Offline: {e}")

else:
    st.markdown("""
        <div style="text-align: center; padding: 50px 0; background: #ffffff; border-radius: 20px; border: 1px dashed #cbd5e1;">
            <div style="font-size: 3rem;">📖</div>
            <h2 style="color: #1e293b;">Your Document Brain is Ready</h2>
            <p style="color: #64748b;">Ask a question above to begin retrieval.</p>
        </div>
    """, unsafe_allow_html=True)

