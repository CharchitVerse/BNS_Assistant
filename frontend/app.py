"""
BNS Legal RAG — Streamlit Frontend
====================================
Chat interface for querying Bharatiya Nyaya Sanhita 2023.

Features:
- Chat UI with streaming responses
- Section browser sidebar
- Feedback thumbs (👍/👎)
- Conversation history
- Section detail viewer

Deploy: Streamlit Community Cloud (free)
"""

import json
import uuid
from datetime import datetime

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except (FileNotFoundError, KeyError):
    BACKEND_URL = "http://localhost:8000"

API_TIMEOUT = 30.0

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BNS Legal AI — Bharatiya Nyaya Sanhita",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a0f;
    }

    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    .assistant-message {
        background: linear-gradient(135deg, #0f1923, #121a28);
        border: 1px solid #1e3a4a;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Section citation cards */
    .section-card {
        background: #12121a;
        border: 1px solid #2a2a3a;
        border-left: 3px solid #4ecdc4;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        font-size: 13px;
    }

    .section-number {
        color: #4ecdc4;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Confidence badge */
    .confidence-high { color: #4ecdc4; }
    .confidence-medium { color: #ffd93d; }
    .confidence-low { color: #ff6b6b; }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        color: #ff6b6b;
        margin-top: 10px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .sidebar-section {
        background: #12121a;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 10px;
        margin: 6px 0;
        cursor: pointer;
    }

    .sidebar-section:hover {
        border-color: #4ecdc4;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = None


init_session_state()


# ── API Client ────────────────────────────────────────────────────────────────


# class BNSApiClient:
#     """HTTP client for the FastAPI backend."""
#
#     def __init__(self, base_url: str = BACKEND_URL):
#         self.base_url = base_url.rstrip("/")
#         self.client = httpx.AsyncClient(timeout=API_TIMEOUT)
#
#     async def query(self, question: str, session_id: str, top_k: int = 5) -> dict:
#         """Send a query to the backend."""
#         response = await self.client.post(
#             f"{self.base_url}/api/query",
#             json={
#                 "query": question,
#                 "session_id": session_id,
#                 "top_k": top_k,
#             },
#         )
#         response.raise_for_status()
#         return response.json()
#
#     async def get_section(self, section_number: str) -> dict:
#         """Get details of a specific section."""
#         response = await self.client.get(
#             f"{self.base_url}/api/sections/{section_number}"
#         )
#         response.raise_for_status()
#         return response.json()
#
#     async def submit_feedback(self, message_id: str, rating: int, comment: str = None):
#         """Submit feedback on a response."""
#         response = await self.client.post(
#             f"{self.base_url}/api/feedback",
#             json={
#                 "message_id": message_id,
#                 "rating": rating,
#                 "comment": comment,
#             },
#         )
#         response.raise_for_status()
#
#     async def health_check(self) -> dict:
#         """Check backend health."""
#         try:
#             response = await self.client.get(f"{self.base_url}/api/health")
#             return response.json()
#         except Exception:
#             return {"status": "unreachable"}


# Synchronous wrapper for Streamlit
def query_sync(question: str, session_id: str, top_k: int = 5) -> dict:
    """Sync wrapper using httpx sync client."""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/query",
            json={"query": question, "session_id": session_id, "top_k": top_k},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_section_sync(section_number: str) -> dict:
    try:
        response = httpx.get(f"{BACKEND_URL}/api/sections/{section_number}", timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def submit_feedback_sync(message_id: str, rating: int):
    try:
        httpx.post(f"{BACKEND_URL}/api/feedback", json={"message_id": message_id, "rating": rating}, timeout=API_TIMEOUT)
    except Exception:
        pass


def health_check_sync() -> dict:
    try:
        response = httpx.get(f"{BACKEND_URL}/api/health", timeout=5.0)
        return response.json()
    except Exception:
        return {"status": "unreachable"}


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚖️ BNS Legal AI")
        st.markdown("*Bharatiya Nyaya Sanhita 2023*")
        st.divider()

        # Health status
        health = health_check_sync()
        status = health.get("status", "unknown")
        status_emoji = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
        st.markdown(f"**Status:** {status_emoji} {status}")

        if health.get("vector_store"):
            st.caption(f"Vector Store: {health['vector_store']}")

        st.divider()

        # Settings
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Sections to retrieve", 1, 10, 5, key="top_k_slider")

        st.divider()

        # Quick section lookup
        st.markdown("### 📖 Section Lookup")
        section_num = st.text_input("Enter section number (1-358)", key="section_lookup")
        if section_num and section_num.isdigit():
            section_data = get_section_sync(section_num)
            if section_data:
                st.markdown(f"**Section {section_data['section_number']}:** {section_data['section_title']}")
                st.markdown(f"*{section_data['chapter']}*")
                with st.expander("Full Text"):
                    st.markdown(section_data["full_text"][:2000])
                if section_data.get("punishment"):
                    st.warning(f"**Punishment:** {section_data['punishment'][:300]}")
            else:
                st.error("Section not found")

        st.divider()

        # New conversation
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        # Example queries
        st.markdown("### 💡 Try asking")
        examples = [
            "What is the punishment for murder under BNS?",
            "Explain Section 69 about sexual offences",
            "What are the provisions for cyber crime?",
            "Compare theft and robbery under BNS",
            "What constitutes criminal conspiracy?",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

        st.divider()
        st.caption("⚠️ AI-generated legal info, not legal advice.")
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")


# ── Chat Display ──────────────────────────────────────────────────────────────


def display_message(msg: dict):
    """Display a single chat message with metadata."""
    role = msg["role"]

    with st.chat_message(role, avatar="👤" if role == "user" else "⚖️"):
        st.markdown(msg["content"])

        # Show cited sections for assistant messages
        if role == "assistant" and msg.get("cited_sections"):
            with st.expander(f"📄 Cited Sections ({len(msg['cited_sections'])})"):
                for cs in msg["cited_sections"]:
                    st.markdown(
                        f"""<div class="section-card">
                        <span class="section-number">Section {cs['section_number']}</span>: 
                        {cs['section_title']}<br>
                        <small>Relevance: {cs['similarity_score']:.2%}</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Show metadata
        if role == "assistant" and msg.get("metadata"):
            meta = msg["metadata"]
            cols = st.columns(4)
            with cols[0]:
                confidence = meta.get("confidence_score", 0)
                color = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
                st.markdown(f'<span class="{color}">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
            with cols[1]:
                st.caption(f"⏱️ {meta.get('latency_ms', 0):.0f}ms")
            with cols[2]:
                st.caption(f"🤖 {meta.get('model_used', 'unknown')}")
            with cols[3]:
                if meta.get("cached"):
                    st.caption("💾 Cached")

        # Feedback buttons
        if role == "assistant" and msg.get("message_id"):
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{msg['message_id']}"):
                    submit_feedback_sync(msg["message_id"], 1)
                    st.toast("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"down_{msg['message_id']}"):
                    submit_feedback_sync(msg["message_id"], -1)
                    st.toast("Thanks, we'll improve!")


# ── Main Chat Interface ──────────────────────────────────────────────────────


def main():
    render_sidebar()

    # Header
    st.markdown("""
    # ⚖️ BNS Legal AI
    ### Query the Bharatiya Nyaya Sanhita 2023 with AI
    """)

    st.markdown(
        '<div class="disclaimer">⚠️ This is an AI-powered tool for informational purposes only. '
        'It does not constitute legal advice. Always consult a qualified lawyer for legal matters.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg)

    # Handle pending query from sidebar examples
    pending = st.session_state.pop("pending_query", None)

    # Chat input
    user_input = st.chat_input("Ask about any BNS provision, offence, or punishment...")
    query = pending or user_input

    if query:
        # Add user message
        user_msg = {"role": "user", "content": query}
        st.session_state.messages.append(user_msg)
        display_message(user_msg)

        # Get response
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Searching BNS sections and generating answer..."):
                top_k = st.session_state.get("top_k_slider", 5)
                result = query_sync(query, st.session_state.session_id, top_k)

            if "error" in result:
                st.error(f"Error: {result['error']}")
                assistant_msg = {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {result['error']}. Please try again.",
                }
            else:
                # Display answer
                st.markdown(result["answer"])

                # Show citations
                if result.get("cited_sections"):
                    with st.expander(f"📄 Cited Sections ({len(result['cited_sections'])})"):
                        for cs in result["cited_sections"]:
                            st.markdown(
                                f"""<div class="section-card">
                                <span class="section-number">Section {cs['section_number']}</span>: 
                                {cs['section_title']}<br>
                                <small>Relevance: {cs['similarity_score']:.2%}</small>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                # Metadata row
                cols = st.columns(4)
                with cols[0]:
                    conf = result.get("confidence_score", 0)
                    color = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.6 else "confidence-low"
                    st.markdown(f'<span class="{color}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                with cols[1]:
                    st.caption(f"⏱️ {result.get('latency_ms', 0):.0f}ms")
                with cols[2]:
                    st.caption(f"🤖 {result.get('model_used', '')}")
                with cols[3]:
                    if result.get("cached"):
                        st.caption("💾 Cached")

                # Feedback
                msg_id = result.get("message_id", str(uuid.uuid4()))
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{msg_id}"):
                        submit_feedback_sync(msg_id, 1)
                        st.toast("Thanks!")
                with col2:
                    if st.button("👎", key=f"down_{msg_id}"):
                        submit_feedback_sync(msg_id, -1)
                        st.toast("Thanks!")

                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"],
                    "cited_sections": result.get("cited_sections", []),
                    "message_id": msg_id,
                    "metadata": {
                        "confidence_score": result.get("confidence_score", 0),
                        "latency_ms": result.get("latency_ms", 0),
                        "model_used": result.get("model_used", ""),
                        "cached": result.get("cached", False),
                    },
                }

            st.session_state.messages.append(assistant_msg)


if __name__ == "__main__":
    main()
