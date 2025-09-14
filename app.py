import streamlit as st
from backend import build_graph, cleanup_recommendation_agents, get_cache_stats
import os

st.set_page_config(page_title="Music Store Chatbot", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stApp * {
        color: #ffffff !important;
    }
    .stChatMessage.user {
        background-color: #2a2a2a;
        color: #ffffff;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 5px 0;
        text-align: left;
    }
    .stChatMessage.assistant {
        background-color: #444;
        color: #ffffff;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 5px 0;
        text-align: right;
    }
     /* Keep chat input readable - multiple selectors */
    .stChatInput input,
    .stChatInput textarea,
    input[data-testid="stChatInput"],
    textarea[data-testid="stChatInput"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Music Store Chatbot")
st.write("Welcome! Chat with our intelligent music store assistant.")

# Environment variables
# Environment variables for API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize app state
if "app" not in st.session_state:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    st.session_state.app = build_graph()
    st.session_state.customer_id = "CUST4847"
    st.session_state.history = []
    st.session_state.last_agent = ""
    st.session_state.router_conversation_id = ""

# Show chat history
for msg in st.session_state.history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# Chat input at bottom
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with graph
    inputs = {
        "query": user_input,
        "customer_id": st.session_state.customer_id,
        "history": [h[1] for h in st.session_state.history],
        "last_agent": st.session_state.last_agent,
        "router_conversation_id": st.session_state.router_conversation_id,
    }
    final_state = st.session_state.app.invoke(inputs)
    agent_response = final_state["response"]
    handled_by = final_state.get("last_agent", "general_agent")
    st.session_state.router_conversation_id = final_state.get("router_conversation_id", "")

    # Add agent reply
    st.session_state.history.append(("assistant", agent_response))
    with st.chat_message("assistant"):
        st.markdown(agent_response)

    st.session_state.last_agent = handled_by

# Cache stats
if st.button("📊 Show Cache Stats"):
    stats = get_cache_stats()
    st.write(stats)

# Cleanup
def on_session_end():
    cleanup_recommendation_agents()
    st.write("Cleanup completed.")

st.session_state.on_session_end = on_session_end


