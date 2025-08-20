# app.py
# PURPOSE: Web UI for your chatbot that:
# 1) Auto-loads a CSV at startup from the project folder (no prompt)
# 2) Lets you drag-and-drop a CSV; it reloads immediately (no extra clicks)
# 3) Disables chat until KB is ready
# 4) Uses HFSmartCSVChatbot (embeddings + TF-IDF + classic hybrid) for better matches

from __future__ import annotations
import os
import re
import tempfile
from glob import glob

import streamlit as st
from hf_retrieval import HFSmartCSVChatbot

st.set_page_config(page_title="IT Support Chatbot", page_icon="💬", layout="wide")
st.title("💬 IT Support Chatbot")

# ---------- helpers ----------
def ensure_bot() -> HFSmartCSVChatbot:
    """Create/return a singleton HFSmartCSVChatbot stored in session_state."""
    if "bot" not in st.session_state:
        bot = HFSmartCSVChatbot()
        # sensible defaults
        bot.use_hybrid = True
        bot.use_embeddings = True
        bot.use_cross_encoder = False
        bot._top_k = 5
        bot._emb_weight = 0.6      # embeddings weight
        bot._tfidf_weight = 0.25   # TF-IDF weight
        bot._min_conf = 0.25
        st.session_state.bot = bot
    return st.session_state.bot

def write_temp_csv(uploaded_file) -> str:
    """Persist uploaded CSV to a stable path in this session."""
    if "csv_tmp_path" not in st.session_state:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        st.session_state.csv_tmp_path = tmp.name
        tmp.close()
    with open(st.session_state.csv_tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.session_state.csv_tmp_path

def csv_has_qa_columns(path: str) -> bool:
    """Quick header check for Question/Answer columns (case-insensitive)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = f.readline()
        cols = [c.strip().lower() for c in re.split(r"[,\t;|]", header)]
        has_q = any(c in ("question", "questions", "q", "query") for c in cols)
        has_a = any(c in ("answer", "answers", "a", "response", "solution") for c in cols)
        return has_q and has_a
    except Exception:
        return False

def find_default_csv() -> str | None:
    """
    Find a CSV to auto-load in the project folder:
      1) common names next to this file
      2) any *.csv in this folder that has Question/Answer headers
    """
    base = os.path.dirname(__file__)
    # 1) common names first
    for name in ("school_it_qa.csv", "it_qa.csv", "questions.csv", "data.csv"):
        p = os.path.join(base, name)
        if os.path.exists(p) and csv_has_qa_columns(p):
            return p
    # 2) any CSV with Q/A columns
    for p in sorted(glob(os.path.join(base, "*.csv"))):
        if csv_has_qa_columns(p):
            return p
    return None

def bootstrap_kb(bot: HFSmartCSVChatbot):
    """
    Load a CSV automatically in this order:
      1) If a CSV was uploaded this session, load it.
      2) If a previously saved config path exists, load it.
      3) Try to auto-load from project folder (common names, then any CSV w/ Q&A headers).
    Sets st.session_state.kb_ready = True/False
    """
    if st.session_state.get("kb_ready") is True:
        return

    # 1) Use freshly uploaded file if present
    if "csv_tmp_path" in st.session_state and os.path.exists(st.session_state.csv_tmp_path):
        if bot.load_csv_file(st.session_state.csv_tmp_path, save_to_config=True):
            st.session_state.kb_ready = True
            st.session_state.loaded_csv = st.session_state.csv_tmp_path
            st.toast("CSV loaded from upload")
            return

    # 2) Try last-used path (saved by bot.save_config)
    try:
        last_path = bot.load_config()
        if last_path and os.path.exists(last_path) and csv_has_qa_columns(last_path):
            if bot.load_csv_file(last_path, save_to_config=False):
                st.session_state.kb_ready = True
                st.session_state.loaded_csv = last_path
                st.toast(f"Loaded last-used CSV: {os.path.basename(last_path)}")
                return
    except Exception:
        pass

    # 3) Auto-discover from project folder
    auto = find_default_csv()
    if auto and bot.load_csv_file(auto, save_to_config=True):
        st.session_state.kb_ready = True
        st.session_state.loaded_csv = auto
        st.toast(f"Auto-loaded KB: {os.path.basename(auto)}")
        return

    st.session_state.kb_ready = False
    st.session_state.loaded_csv = None

# Initialize once (AFTER helpers are defined)
bot = ensure_bot()
bootstrap_kb(bot)

# Show which CSV is loaded (if any)
if st.session_state.get("kb_ready") and st.session_state.get("loaded_csv"):
    st.caption(f"Using KB: **{os.path.basename(st.session_state.loaded_csv)}**")

# ---------- sidebar: upload + settings ----------
with st.sidebar:
    st.header("📄 Knowledge Base")

    # Drag-and-drop CSV anytime; the app will reload and start using it immediately.
    up = st.file_uploader("Drag & drop a CSV (Question & Answer columns)", type=["csv"])
    if up is not None:
        csv_path = write_temp_csv(up)
        if bot.load_csv_file(csv_path, save_to_config=True):
            st.success("CSV loaded!")
            st.session_state.kb_ready = True
            st.session_state.loaded_csv = csv_path
            st.session_state.messages = [{"role": "assistant", "content": "KB loaded. Ask me an IT question."}]
            st.rerun()
        else:
            st.error("Failed to load CSV. Ensure it has 'Question' and 'Answer' columns.")

    st.divider()
    st.subheader("🎯 Context Filters")
    meta = bot.get_meta_vocab() if hasattr(bot, "get_meta_vocab") else {"categories": [], "school_levels": []}
    cats = st.multiselect("Category", meta["categories"])
    lvls = st.multiselect("School Level", meta["school_levels"])
    strict = st.toggle("Strict filter (only show matches in selected filters)", False)
    bot.set_context(categories=cats or None, school_levels=lvls or None, strict_category_filter=strict)

    st.divider()
    st.subheader("🧠 Retrieval Settings")
    bot.use_hybrid = st.toggle("Enable hybrid retrieval", True)

    # 🤗 Hugging Face controls (embeddings + optional cross-encoder)
    st.subheader("🤗 Hugging Face")
    bot.use_embeddings = st.toggle("Use embeddings (SentenceTransformers)", True)
    bot._emb_weight = st.slider("Embedding weight", 0.0, 1.0, bot._emb_weight, 0.05)
    bot.use_cross_encoder = st.toggle("Use cross-encoder re-ranker (slower)", False)
    st.caption("Models: all-MiniLM-L6-v2 (emb), ms-marco-MiniLM-L-6-v2 (re-rank)")

    # TF-IDF + acceptance threshold
    st.subheader("🔎 TF-IDF & Threshold")
    bot._tfidf_weight = st.slider("TF-IDF weight", 0.0, 1.0, bot._tfidf_weight, 0.05)
    classic_weight = max(0.0, 1.0 - bot._emb_weight - bot._tfidf_weight)
    st.caption(f"Classic (Jaccard+keywords) weight = 1 − emb − tfidf = **{classic_weight:.2f}**")
    bot._top_k = st.slider("Top-K candidates", 3, 10, bot._top_k)
    bot._min_conf = st.slider("Min confidence to accept answer", 0.0, 1.0, bot._min_conf, 0.01)

    st.divider()
    st.subheader("📊 KB Stats")
    total = len(bot.knowledge_base)
    st.metric("Total Q&A pairs", total)
    if total > 0:
        examples = []
        for i, (_, data) in enumerate(list(bot.knowledge_base.items())[:5], 1):
            q = data["original_question"]
            examples.append(f"{i}. {q if len(q) < 120 else q[:117] + '...'}")
        st.caption("Samples:")
        st.write("\n".join(examples))

    st.divider()
    if st.button("Reset chat"):
        st.session_state.messages = (
            [{"role": "assistant", "content": "KB loaded. Ask me an IT question."}]
            if st.session_state.get("kb_ready") else []
        )
        st.rerun()

# ---------- chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = (
        [{"role": "assistant", "content": "KB loaded. Ask me an IT question."}]
        if st.session_state.get("kb_ready") else []
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------- chat input (disabled until KB ready) ----------
disabled = not st.session_state.get("kb_ready", False)
placeholder = "Upload a CSV to start" if disabled else "Type your question…"
user_text = st.chat_input(placeholder=placeholder, disabled=disabled)

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    # Sentiment badge (so you can “see” the brain)
    sentiment = bot.sentiment_analyzer.analyze_sentiment(user_text)
    sent_badge = (
        f"Sentiment: **{sentiment['sentiment']}** • "
        f"Urgent: **{sentiment['is_urgent']}** • "
        f"Frustrated: **{sentiment['is_frustrated']}** • "
        f"Confused: **{sentiment['is_confused']}** • "
        f"Intensity: **{sentiment['intensity']:.2f}**"
    )

    # Answer
    reply = bot.get_response(user_text)

    with st.chat_message("assistant"):
        st.markdown(reply)
        st.caption(sent_badge)

        # Diagnostics (why this answer)
        if bot.use_hybrid:
            cands = bot._hybrid_candidates(user_text)
            if cands:
                top = cands[0][1]
                st.progress(min(max(top, 0.0), 1.0), text=f"Confidence ~ {top:.2f}")
                with st.expander("Why this answer? (top candidates)"):
                    for ans, score, dbg in cands[:bot._top_k]:
                        qsrc = dbg.get("q", "KB")
                        st.write(f"**Score:** {score:.3f}  •  **Matched question:** {qsrc}")
                        st.code((ans[:500] + "…") if len(ans) > 500 else ans)

    st.session_state.messages.append({"role": "assistant", "content": reply})
