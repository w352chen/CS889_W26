"""
Literature Review Helper - Compact UI with example-bib.json only.
Sidebar: Brainstorm with AI chatbot.
"""

import os
import json
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Literature Review Helper", layout="wide")

# Much smaller fonts + very compact layout (minimal scrolling)
st.markdown("""
<style>
    /* Significantly reduce font sizes */
    .stApp, .block-container { font-size: 0.75rem !important; }
    h1 { font-size: 1.2rem !important; margin-bottom: 0.3rem !important; }
    h2, h3 { font-size: 0.95rem !important; margin-bottom: 0.2rem !important; }
    .stMarkdown, p { font-size: 0.75rem !important; line-height: 1.3 !important; }
    .stTextInput label, .stTextArea label, .stSelectbox label { font-size: 0.7rem !important; }
    .stButton button { font-size: 0.7rem !important; padding: 0.25rem 0.5rem !important; }
    .stCheckbox label { font-size: 0.7rem !important; }
    
    /* Compact sidebar - hide collapse button, move content up */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 320px !important;
        padding-top: 0 !important;
    }
    
    /* Move sidebar content up - minimal top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.25rem !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
    /* Very tight spacing */
    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 0.5rem !important; 
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stExpander { margin-bottom: 0.2rem !important; }
    div[data-testid="stExpander"] details summary { 
        padding: 0.3rem 0.5rem !important; 
        font-size: 0.72rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    div[data-testid="stExpander"] div[role="region"] { padding: 0.3rem 0.5rem !important; }
    
    /* Fix expander title wrapping */
    div[data-testid="stExpander"] summary p {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Smaller chat messages */
    [data-testid="stChatMessage"] { 
        padding: 0.35rem 0.6rem !important; 
        margin-bottom: 0.3rem !important;
        font-size: 0.72rem !important;
    }
    [data-testid="stChatMessageContent"] { font-size: 0.72rem !important; }
    
    /* Compact input fields */
    .stTextInput input, .stTextArea textarea { 
        font-size: 0.75rem !important; 
        padding: 0.3rem 0.5rem !important;
    }
    
    /* Reduce tab spacing */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem !important; }
    .stTabs [data-baseweb="tab"] { 
        padding: 0.3rem 0.8rem !important; 
        font-size: 0.75rem !important;
    }
    
    /* Compact columns */
    [data-testid="column"] { padding: 0.2rem !important; }
    
    /* Hide Streamlit menu and unnecessary buttons */
    #MainMenu { visibility: hidden !important; }
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stActionButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
BIB_PATH = "example-bib.json"

# -----------------------------
# Session state
# -----------------------------
def ensure_session_state():
    if "selected" not in st.session_state:
        st.session_state.selected = {}
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = pd.DataFrame()
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "brainstorm_history" not in st.session_state:
        st.session_state.brainstorm_history = []  # [{"role": "user"/"assistant", "content": "..."}]
    if "ai_keyword_suggestions" not in st.session_state:
        st.session_state.ai_keyword_suggestions = []
    if "ai_search_strings" not in st.session_state:
        st.session_state.ai_search_strings = []


ensure_session_state()


# -----------------------------
# Data loading (example-bib.json only)
# -----------------------------
@st.cache_data
def load_bib(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rows = data.get("references", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    for col in ["title", "abstract", "venue", "url", "type"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    
    if "journal" in df.columns and "venue" not in df.columns:
        df["venue"] = df["journal"].fillna("").astype(str)
    
    if "year" not in df.columns:
        df["year"] = 0
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    
    if "authors" in df.columns:
        df["authors"] = df["authors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    else:
        df["authors"] = ""
    
    return df


def local_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df
    mask = (
        df["title"].str.lower().str.contains(q, na=False)
        | df["abstract"].str.lower().str.contains(q, na=False)
        | df["authors"].str.lower().str.contains(q, na=False)
        | df["venue"].str.lower().str.contains(q, na=False)
    )
    return df[mask].copy()


# -----------------------------
# Gemini helpers
# -----------------------------
@st.cache_resource
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def gemini_generate(prompt: str, model: str = DEFAULT_GEMINI_MODEL) -> str:
    client = get_gemini_client(GEMINI_API_KEY)
    if client is None:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


def gemini_suggest_keywords(topic: str, rq: str) -> Tuple[List[str], List[str]]:
    prompt = f"""
You help researchers find literature. Given topic and optional research question, output JSON:
{{"keywords": ["keyword1", ...], "search_strings": ["search1", ...]}}

Topic: {topic}
Research question: {rq or "(none)"}

Return ONLY valid JSON."""

    raw = gemini_generate(prompt)
    try:
        data = json.loads(raw)
        kws = [str(x).strip() for x in data.get("keywords", []) if str(x).strip()][:20]
        ss = [str(x).strip() for x in data.get("search_strings", []) if str(x).strip()][:10]
        return kws, ss
    except Exception:
        lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
        return lines[:15], [ln for ln in lines if "AND" in ln.upper() or '"' in ln][:5]


def gemini_brainstorm(user_msg: str, history: List[dict]) -> str:
    ctx = "\n".join([f"{h['role']}: {h['content'][:500]}" for h in history[-6:]])
    prompt = f"""You are a literature review assistant. Help the user brainstorm: keywords, search strategies, research angles.
Keep replies concise (2–4 sentences). If they ask for keywords, suggest 5–8 specific terms.

Previous context:
{ctx or "(none)"}

User: {user_msg}

Assistant:"""
    return gemini_generate(prompt)


def gemini_rank_results(topic: str, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    sample = results.head(12)
    items = [{"id": int(i), "title": str(r.get("title", ""))[:300], "abstract": str(r.get("abstract", ""))[:600]}
            for i, r in sample.iterrows()]
    prompt = f"""For literature review on "{topic}", label each paper: "Highly relevant", "Maybe", or "Not relevant". One-sentence reason.
Return JSON: {{"rankings": [{{"id":0,"label":"...","reason":"..."}}, ...]}}
Papers: {json.dumps(items, ensure_ascii=False)}"""
    raw = gemini_generate(prompt)
    try:
        data = json.loads(raw)
        rankings = {int(r["id"]): (r.get("label", ""), r.get("reason", "")) for r in data.get("rankings", [])}
    except Exception:
        rankings = {}
    out = results.copy()
    out["ai_label"], out["ai_reason"] = "", ""
    for i in sample.index:
        lbl, reason = rankings.get(int(i), ("", ""))
        out.at[i, "ai_label"], out.at[i, "ai_reason"] = lbl, reason
    return out


# -----------------------------
# SIDEBAR: Brainstorm with AI (ChatGPT-style)
# -----------------------------
with st.sidebar:
    st.markdown("## 💡 Brainstorm with AI")
    
    # Research question input
    with st.expander("🎯 Research Setup", expanded=True):
        topic_input = st.text_input(
            "Topic",
            placeholder="e.g., cognitive drift in knowledge systems",
            key="topic_input"
        )
        rq_input = st.text_area(
            "Research question (optional)",
            placeholder="What aspects do you want to explore?",
            height=60,
            key="rq_input"
        )
        
        if st.button("✨ Get Keywords from AI", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("Missing GEMINI_API_KEY in .env")
            elif topic_input.strip():
                with st.spinner("Generating keywords..."):
                    try:
                        kws, search_strings = gemini_suggest_keywords(topic_input.strip(), rq_input.strip())
                        st.session_state.ai_keyword_suggestions = kws
                        st.session_state.ai_search_strings = search_strings
                        st.session_state.brainstorm_history.append({
                            "role": "user",
                            "content": f"Topic: {topic_input}\nRQ: {rq_input or '(none)'}"
                        })
                        st.session_state.brainstorm_history.append({
                            "role": "assistant",
                            "content": f"**Keywords**: {', '.join(kws[:8])}\n\n**Search strings**: See main area for clickable options."
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Enter a topic first")
    
    st.divider()
    
    # ChatGPT-style conversation area
    st.caption("Chat with AI")
    
    # Display chat history
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.brainstorm_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input at bottom
    user_msg = st.chat_input("Ask about keywords, search strategies, research angles...")
    if user_msg:
        st.session_state.brainstorm_history.append({"role": "user", "content": user_msg})
        with st.spinner("Thinking..."):
            try:
                reply = gemini_brainstorm(user_msg, st.session_state.brainstorm_history)
                st.session_state.brainstorm_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.brainstorm_history.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()
    
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.brainstorm_history = []
            st.session_state.ai_keyword_suggestions = []
            st.session_state.ai_search_strings = []
            st.rerun()
    
    st.divider()
    
    # Selected papers count
    n_sel = len(st.session_state.selected)
    st.caption(f"📎 {n_sel} paper{'s' if n_sel != 1 else ''} selected")
    if st.button("Clear selected", use_container_width=True):
        st.session_state.selected = {}
        st.rerun()


# -----------------------------
# Main layout
# -----------------------------
st.markdown("# 📚 Literature Review Helper")

tab_search, tab_chat = st.tabs(["🔎 Search", "💬 Chat"])

# Load data (example-bib.json only)
try:
    df = load_bib(BIB_PATH)
except FileNotFoundError:
    st.error(f"File not found: {BIB_PATH}. Place it next to app.py.")
    st.stop()

# -----------------------------
# Tab 1: Search & Select
# -----------------------------
with tab_search:
    st.session_state.query = st.text_input(
        "Search",
        value=st.session_state.query,
        placeholder='Keywords or phrases...',
        label_visibility="collapsed"
    )
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        only_reviews = st.checkbox("Reviews only", value=False)
    with c2:
        limit = st.slider("Max", 5, 50, 25, label_visibility="collapsed")
    with c3:
        st.caption(f"Max: {limit}")
    
    results = local_filter(df, st.session_state.query)
    
    if only_reviews:
        qmask = (
            results["title"].str.lower().str.contains("survey|review|meta-analysis", na=False, regex=True)
            | results["abstract"].str.lower().str.contains("survey|review|meta-analysis", na=False, regex=True)
        )
        results = results[qmask].copy()
    
    results = results[results["abstract"].str.len() > 0].head(limit).reset_index(drop=True)
    
    # AI suggestions from brainstorm (keywords as clickable chips)
    if st.session_state.ai_keyword_suggestions or st.session_state.ai_search_strings:
        with st.expander("✨ AI Keyword Suggestions", expanded=True):
            if st.session_state.ai_keyword_suggestions:
                st.caption("Click keywords to add to search:")
                chip_cols = st.columns(6)
                for i, kw in enumerate(st.session_state.ai_keyword_suggestions[:12]):
                    if chip_cols[i % 6].button(kw, key=f"kw_{i}", use_container_width=True):
                        st.session_state.query = (st.session_state.query + " " + kw).strip()
                        st.rerun()
            
            if st.session_state.ai_search_strings:
                st.caption("Click to use search string:")
                for j, ss in enumerate(st.session_state.ai_search_strings[:4]):
                    if st.button(f"🔍 {ss[:55]}{'...' if len(ss) > 55 else ''}", key=f"ss_{j}"):
                        st.session_state.query = ss
                        st.rerun()
    
    # AI rank button (compact)
    col_rank, col_spacer = st.columns([1, 3])
    with col_rank:
        rank_btn = st.button("🧠 AI rank", use_container_width=True)
    
    if rank_btn and not results.empty and GEMINI_API_KEY:
        with st.spinner("Ranking..."):
            try:
                results = gemini_rank_results(st.session_state.query or "literature", results)
                st.session_state.last_results = results.copy()
            except Exception as e:
                st.error(str(e))
    
    st.session_state.last_results = results.copy()
    
    left, right = st.columns([2.5, 1])
    
    with left:
        st.markdown(f"**Results ({len(results)})**")
        
        if results.empty:
            st.info("No results. Try a broader query or use Brainstorm → Get Keywords.")
        else:
            for idx, row in results.iterrows():
                title = row.get("title", "").strip() or f"(untitled {idx})"
                ai_label = row.get("ai_label", "")
                
                # Add AI label badge if available - keep full title, let CSS handle wrapping
                title_display = title
                if ai_label:
                    badge_color = {"Highly relevant": "🟢", "Maybe": "🟡", "Not relevant": "🔴"}.get(ai_label, "⚪")
                    title_display = f"{badge_color} {title_display}"
                
                with st.expander(title_display, expanded=False):
                    st.caption(f"**{row.get('authors','')}** · {row.get('year','')}")
                    
                    # Show AI reason if available
                    if row.get("ai_reason"):
                        st.info(f"💡 AI: {row.get('ai_reason')}", icon="🤖")
                    
                    # Truncated abstract
                    abstract = str(row.get("abstract", ""))
                    st.markdown(abstract[:350] + ("..." if len(abstract) > 350 else ""))
                    
                    # Select checkbox
                    col_sel, col_venue = st.columns([1, 2])
                    with col_sel:
                        selected = st.checkbox("Select", key=f"sel_{idx}", value=title in st.session_state.selected)
                    with col_venue:
                        if row.get("venue"):
                            st.caption(f"📄 {row.get('venue')}")
                    
                    if selected:
                        st.session_state.selected[title] = row.to_dict()
                    elif title in st.session_state.selected:
                        del st.session_state.selected[title]
    
    with right:
        st.markdown("**Selected Papers**")
        if not st.session_state.selected:
            st.caption("None yet")
        else:
            for i, (t, rec) in enumerate(list(st.session_state.selected.items())[:12]):
                col_txt, col_btn = st.columns([4, 1])
                with col_txt:
                    st.caption(f"{i+1}. {t[:40]}...")
                with col_btn:
                    if st.button("✕", key=f"rm_{i}"):
                        del st.session_state.selected[t]
                        st.rerun()


# -----------------------------
# Tab 2: Chat with AI
# -----------------------------
with tab_chat:
    col_inc, col_clr = st.columns([3, 1])
    with col_inc:
        include_papers = st.checkbox("Include selected papers in context", value=True)
    with col_clr:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat = []
            st.rerun()
    
    # Display chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your literature...")
    
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        
        ctx = ""
        if include_papers and st.session_state.selected:
            ctx = "Selected papers:\n" + "\n".join([
                f"- {t}: {r.get('abstract','')[:250]}" 
                for t, r in list(st.session_state.selected.items())[:5]
            ])
        
        try:
            full_prompt = f"{ctx}\n\nUser: {user_input}\n\nAssistant (concise):" if ctx else f"User: {user_input}\n\nAssistant:"
            reply = gemini_generate(full_prompt)
            st.session_state.chat.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()
