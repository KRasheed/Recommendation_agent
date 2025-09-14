# import streamlit as st
# from backend import build_graph, cleanup_recommendation_agents, get_cache_stats
# import os

# st.set_page_config(page_title="Music Store Chatbot", layout="wide")

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #1a1a1a;
#         color: #ffffff;
#     }
#     .stApp * {
#         color: #ffffff !important;
#     }
#     .stChatMessage.user {
#         background-color: #2a2a2a;
#         color: #ffffff;
#         border-radius: 12px;
#         padding: 8px 12px;
#         margin: 5px 0;
#         text-align: left;
#     }
#     .stChatMessage.assistant {
#         background-color: #444;
#         color: #ffffff;
#         border-radius: 12px;
#         padding: 8px 12px;
#         margin: 5px 0;
#         text-align: right;
#     }
#      /* Keep chat input readable - multiple selectors */
#     .stChatInput input,
#     .stChatInput textarea,
#     input[data-testid="stChatInput"],
#     textarea[data-testid="stChatInput"] {
#         color: #000000 !important;
#         background-color: #ffffff !important;
#     </style>
# """, unsafe_allow_html=True)

# st.title("🎵 Music Store Chatbot")
# st.write("Welcome! Chat with our intelligent music store assistant.")

# # Environment variables
# # Environment variables for API keys
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# # Initialize app state
# if "app" not in st.session_state:
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#     st.session_state.app = build_graph()
#     st.session_state.customer_id = "CUST4847"
#     st.session_state.history = []
#     st.session_state.last_agent = ""
#     st.session_state.router_conversation_id = ""

# # Show chat history
# for msg in st.session_state.history:
#     role, text = msg
#     with st.chat_message(role):
#         st.markdown(text)

# # Chat input at bottom
# if user_input := st.chat_input("Type your message..."):
#     # Add user message
#     st.session_state.history.append(("user", user_input))
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Process with graph
#     inputs = {
#         "query": user_input,
#         "customer_id": st.session_state.customer_id,
#         "history": [h[1] for h in st.session_state.history],
#         "last_agent": st.session_state.last_agent,
#         "router_conversation_id": st.session_state.router_conversation_id,
#     }
#     final_state = st.session_state.app.invoke(inputs)
#     agent_response = final_state["response"]
#     handled_by = final_state.get("last_agent", "general_agent")
#     st.session_state.router_conversation_id = final_state.get("router_conversation_id", "")

#     # Add agent reply
#     st.session_state.history.append(("assistant", agent_response))
#     with st.chat_message("assistant"):
#         st.markdown(agent_response)

#     st.session_state.last_agent = handled_by

# # Cache stats
# if st.button("📊 Show Cache Stats"):
#     stats = get_cache_stats()
#     st.write(stats)

# # Cleanup
# def on_session_end():
#     cleanup_recommendation_agents()
#     st.write("Cleanup completed.")

# st.session_state.on_session_end = on_session_end
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
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Music Store Chatbot")
st.write("Welcome! Chat with our intelligent music store assistant.")

# Environment variables for API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# ====== CUSTOMER ID INPUT SECTION ======
st.markdown("### Customer Information")
col1, col2 = st.columns([2, 1])

with col1:
    customer_id_input = st.text_input(
        "Enter your Customer ID (e.g., CUST4847):",
        value=st.session_state.get("customer_id", ""),
        placeholder="CUST4847",
        help="Enter your customer ID to start chatting"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    if st.button("🆕 New Chat Session", type="secondary"):
        start_new_chat()
        st.rerun()

# ====== NEW CHAT FUNCTION ======
def start_new_chat():
    """Complete session reset - clears both backend and frontend"""
    # Backend cleanup
    if "customer_id" in st.session_state and st.session_state.customer_id:
        cleanup_recommendation_agents(st.session_state.customer_id)
    else:
        cleanup_recommendation_agents()  # Clean all if no specific customer
    
    # Frontend cleanup - clear all session state
    st.session_state.history = []
    st.session_state.last_agent = ""
    st.session_state.router_conversation_id = ""
    
    # Keep customer_id if entered, but clear chat
    if customer_id_input.strip():
        st.session_state.customer_id = customer_id_input.strip().upper()
    
    # Force app rebuild with new customer
    if "app" in st.session_state:
        del st.session_state.app
    
    st.success("New chat session started!")

# ====== CUSTOMER ID VALIDATION ======
def is_valid_customer_id(cid):
    """Basic validation for customer ID format"""
    return cid and len(cid.strip()) >= 6 and cid.strip().upper().startswith("CUST")

# Update session state customer_id when input changes
if customer_id_input.strip() and customer_id_input.strip().upper() != st.session_state.get("customer_id", ""):
    if is_valid_customer_id(customer_id_input):
        st.session_state.customer_id = customer_id_input.strip().upper()
    else:
        st.warning("Please enter a valid Customer ID (format: CUST followed by numbers, e.g., CUST4847)")

# ====== APP INITIALIZATION (CONDITIONAL) ======
if "customer_id" in st.session_state and st.session_state.customer_id and is_valid_customer_id(st.session_state.customer_id):
    # Initialize app state only if we have valid customer ID
    if "app" not in st.session_state:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        st.session_state.app = build_graph()
        st.session_state.history = []
        st.session_state.last_agent = ""
        st.session_state.router_conversation_id = ""
    
    # Display current customer info
    st.info(f"Chat session active for: **{st.session_state.customer_id}**")
    
    # ====== CHAT INTERFACE ======
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
    
    # ====== CONTROLS SECTION ======
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Show Cache Stats"):
            stats = get_cache_stats()
            st.json(stats)
    
    with col2:
        if st.button("🧹 Clear Current Session"):
            start_new_chat()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Complete Cleanup"):
            cleanup_recommendation_agents()
            st.session_state.clear()
            st.success("Complete cleanup done! Please refresh the page.")

else:
    # ====== NO VALID CUSTOMER ID STATE ======
    if not customer_id_input.strip():
        st.info("👆 Please enter your Customer ID above to start chatting")
    elif not is_valid_customer_id(customer_id_input):
        st.warning("❌ Invalid Customer ID format. Please use format like: CUST4847")
    
    # Show some example customer IDs or help
    with st.expander("Need help with Customer ID?"):
        st.markdown("""
        **Customer ID Format:**
        - Must start with 'CUST'
        - Followed by 4 or more numbers
        - Examples: CUST4847, CUST1234, CUST5678
        
        **Sample Customer IDs for testing:**
        - CUST4847
        - CUST1234  
        - CUST5678
        """)

# ====== FOOTER INFO ======
st.markdown("---")
st.markdown("**🎵 Music Store AI Assistant** - Powered by LangGraph & OpenAI")
if "customer_id" in st.session_state and st.session_state.customer_id:
    st.caption(f"Current session: {st.session_state.customer_id} | Chat messages: {len(st.session_state.get('history', []))}")


