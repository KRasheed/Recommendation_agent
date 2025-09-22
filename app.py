import streamlit as st
from backend import build_graph, cleanup_recommendation_agents, get_cache_stats
import os
import sys
import io
import threading
import time
import re
from contextlib import redirect_stdout

st.set_page_config(
    page_title="Music Store Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Wider sidebar */
    .css-1d391kg {
        width: 350px !important;
    }
    
    .css-1cypcdb {
        width: 350px !important;
    }
    
    [data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 350px !important;
        min-width: 350px !important;
    }
    
    /* Product info box styling */
    .product-info-box {
        background-color: #f0f8ff;
        border: 1px solid #4a90e2;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling for better visibility */
    .stChatMessage {
        border-radius: 10px;
        margin: 8px 0;
        padding: 8px;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
    }
    
    /* Column styling - removed since using single column */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Environment variables for API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

class StreamlitCapture:
    def __init__(self, console_container):
        self.console_container = console_container
        self.captured_output = []
        self.lock = threading.Lock()
        
        # Map backend print statements to user-friendly messages
        self.log_patterns = {
            r"Router Agent Thinking": "🤖 Analyzing your request to find the best specialist...",
            r"Router decision:.*recommendation_agent": "🎵 Directing you to our music recommendation expert...",
            r"Router decision:.*order_tracking_agent": "📦 Connecting you to order tracking specialist...",
            r"Router decision:.*general_agent": "💬 Routing to general customer support...",
            r"Recommendation Agent thinking": "🎶 Our music expert is analyzing your preferences...",
            r"Loading customer data for:": "👤 Loading your customer profile and purchase history...",
            r"Initializing new recommendation agent": "🎵 Setting up personalized music recommendations...",
            r"Agent initialized for customer": "🎵 Let's find your perfect sound match...",
            r"Session initialized for customer": "🎯 Customizing recommendations based on your music taste...",
            r"Customer type:.*New Customer": "👋 Welcome! Setting up recommendations for new customer...",
            r"Customer type:.*Returning Customer": "🎉 Welcome back! Analyzing your music history...",
            r"Owned products:": "📊 Reviewing your previous purchases...",
            r"Searching for.*owned products": "🔍 Cross-referencing your existing music gear...",
            r"Analysing the detailed features of": "🔬 Deep-diving into product specifications...",
            r"Searching catalog to get relevant instruments": "🎼 Browsing our instrument catalog for perfect matches...",
            r"Order Tracking Agent Thinking": "📋 Looking up your order information...",
            r"General Agent Thinking": "💭 Processing your general inquiry...",
            r"Using cached.*data": "⚡ Quickly retrieving your saved preferences...",
            r"Cached.*customer data": "💾 Your preferences have been saved for faster service...",
            r"Error.*recommendation agent": "⚠️ Having trouble with recommendations, working on it...",
            r"Error.*processing": "⚠️ Encountered an issue, finding alternative solution...",
            r".*": "🤖 Processing your request..."
        }
    
    def translate_log(self, text):
        """Convert technical log to user-friendly message"""
        if len(text.strip()) < 5:
            return None
            
        for pattern, message in self.log_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return message
        return "🤖 Processing your request..."
    
    def write(self, text):
        if text.strip():
            with self.lock:
                self.captured_output.append(text.strip())
                
                # Translate and display user-friendly message  
                user_message = self.translate_log(text.strip())
                if user_message and user_message != getattr(self, 'last_shown_message', ''):
                    # Avoid showing duplicate messages
                    self.last_shown_message = user_message
                    self.console_container.markdown(user_message)
                    time.sleep(0.3)  # Brief pause to show progression
    
    def flush(self):
        pass
    
    def clear(self):
        with self.lock:
            self.captured_output = []
            self.console_container.empty()


def is_valid_customer_id(text):
    """
    Validate customer ID format
    - Should start with 'CUST' 
    - Followed by numbers
    - Total length reasonable (e.g., 8-12 characters)
    """
    pattern = r'^CUST\d{4,8}$'  # CUST followed by 4-8 digits
    return bool(re.match(pattern, text.upper().strip()))


def extract_customer_id(text):
    """Extract and format customer ID from user input"""
    return text.upper().strip()


def fetch_customer_products_on_validation(customer_id):
    """
    One-time function to fetch customer products immediately after ID validation.
    Separate from normal chat flow.
    """
    try:
        # Initialize app if needed
        if "app" not in st.session_state:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            st.session_state.app = build_graph()
        
        # Send hardcoded query with proper inputs
        auto_query_inputs = {
            "query": "what products did I order",
            "customer_id": customer_id,
            "history": [],
            "last_agent": "",
            "router_conversation_id": "",
        }
        
        # Invoke backend
        auto_response = st.session_state.app.invoke(auto_query_inputs)
        
        # Extract product info from response
        product_info = auto_response.get("response", "Unable to fetch product information.")
        
        # Update any necessary session state for continuity
        if "router_conversation_id" in auto_response:
            st.session_state.router_conversation_id = auto_response.get("router_conversation_id", "")
        
        return product_info
        
    except Exception as e:
        return f"Error fetching product information: {str(e)}"


def format_product_info(product_info):
    """Format the product information for better display"""
    return product_info


def start_new_chat():
    """Complete session reset - clears both backend and frontend"""
    # Backend cleanup
    cleanup_recommendation_agents()
    
    # Frontend cleanup
    st.session_state.history = []
    st.session_state.last_agent = ""
    st.session_state.router_conversation_id = ""
    
    # Reset customer ID states
    st.session_state.customer_id = ""
    st.session_state.customer_id_validated = False
    st.session_state.waiting_for_customer_id = True
    
    # Add initial bot message
    initial_msg = "Hello! Please enter your Customer ID to start chatting."
    st.session_state.history.append(("assistant", initial_msg))


# ====== SESSION STATE INITIALIZATION ======
if "customer_id" not in st.session_state:
    st.session_state.customer_id = ""
if "customer_id_validated" not in st.session_state:
    st.session_state.customer_id_validated = False
if "waiting_for_customer_id" not in st.session_state:
    st.session_state.waiting_for_customer_id = True
if "history" not in st.session_state:
    st.session_state.history = []
if "last_agent" not in st.session_state:
    st.session_state.last_agent = ""
if "router_conversation_id" not in st.session_state:
    st.session_state.router_conversation_id = ""

# Show initial bot message if history is empty
if len(st.session_state.history) == 0:
    initial_msg = "Hello! Please enter your Customer ID to start chatting."
    st.session_state.history.append(("assistant", initial_msg))

# ====== MAIN UI LAYOUT ======
st.title("🎵 Music Store Chatbot")
st.write("Welcome! Chat with our intelligent music store assistant.")

# Sidebar with New Chat button
st.sidebar.markdown("## Controls")
if st.sidebar.button("🔄 New Chat", type="primary"):
    start_new_chat()
    st.rerun()

# Show current customer ID if validated
if st.session_state.customer_id_validated:
    st.info(f"**Customer ID:** {st.session_state.customer_id}")

# Display chat history
for msg in st.session_state.history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text, unsafe_allow_html=True)

# ====== CHAT INPUT PROCESSING ======
if user_input := st.chat_input("Type your message..."):
    
    # Add user message immediately to history and display
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Case 1: Waiting for Customer ID (first message or after new chat)
    if st.session_state.waiting_for_customer_id:
        
        # Validate the input as customer ID
        # if is_valid_customer_id(user_input):
        #     customer_id = extract_customer_id(user_input)
            
        #     # Save customer ID
        #     st.session_state.customer_id = customer_id
        #     st.session_state.customer_id_validated = True
        #     st.session_state.waiting_for_customer_id = False
            
        #     # Show processing message
        #     with st.chat_message("assistant"):
        #         processing_placeholder = st.empty()
        #         processing_placeholder.markdown("🔍 Validating Customer ID and fetching your order history...")
            
        #     # NEW: Fetch customer products using separate function
        #     product_info = fetch_customer_products_on_validation(customer_id)
            
        #     # Create combined confirmation message with product information
        #     confirmation_msg = f"""Great! Customer ID **{customer_id}** has been validated.
        # Case 1: Waiting for Customer ID (first message or after new chat)
      
            
        # Validate the input as customer ID
        if is_valid_customer_id(user_input):
            customer_id = extract_customer_id(user_input)
            
            # Save customer ID
            st.session_state.customer_id = customer_id
            st.session_state.customer_id_validated = True
            st.session_state.waiting_for_customer_id = False
            
            # Create processing placeholder in main area (not in chat bubble)
            processing_placeholder = st.empty()
            processing_placeholder.markdown("🔍 Validating Customer ID and fetching your order history...")
            
            # Fetch customer products using separate function
            product_info = fetch_customer_products_on_validation(customer_id)
            
            # Create combined confirmation message with product information
            confirmation_msg = f"""Great! Customer ID **{customer_id}** has been validated.


<div class="product-info-box">
📦 {product_info}
</div>

How can I help you today?"""
            
            # Clear processing message and show final confirmation
            processing_placeholder.empty()
            
            # Add combined message to history and display
            st.session_state.history.append(("assistant", confirmation_msg))
            with st.chat_message("assistant"):
                st.markdown(confirmation_msg, unsafe_allow_html=True)
            
            # Initialize app state now that we have valid customer ID (if not already done)
            if "app" not in st.session_state:
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                st.session_state.app = build_graph()
        
        else:
            # Invalid customer ID format
            error_msg = "❌ Invalid Customer ID format. Please enter a valid ID (e.g., CUST1234)."
            st.session_state.history.append(("assistant", error_msg))
            with st.chat_message("assistant"):
                st.markdown(error_msg)
    else:
        # Initialize app if not already done
        if "app" not in st.session_state:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            st.session_state.app = build_graph()
        
        # Create processing placeholder in main area (not in chat bubble)
        processing_placeholder = st.empty()
        
        # Create capture object for processing logs
        capture = StreamlitCapture(processing_placeholder)
        
        # Process with graph while capturing stdout (ALL OUTSIDE CHAT CONTAINER)
        inputs = {
            "query": user_input,
            "customer_id": st.session_state.customer_id,
            "history": [h[1] for h in st.session_state.history[:-1]],
            "last_agent": st.session_state.last_agent,
            "router_conversation_id": st.session_state.router_conversation_id,
        }
        
        # Capture stdout and redirect to processing placeholder
        original_stdout = sys.stdout
        sys.stdout = capture
        
        try:
            final_state = st.session_state.app.invoke(inputs)
            agent_response = final_state["response"]
            handled_by = final_state.get("last_agent", "general_agent")
            st.session_state.router_conversation_id = final_state.get("router_conversation_id", "")
        except Exception as e:
            agent_response = f"Sorry, I encountered an error: {str(e)}"
            handled_by = "error_handler"
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
        
        # Now display final response in chat message container
        with st.chat_message("assistant"):
            processing_placeholder.empty()  # Clear processing logs
            st.markdown(agent_response)     # Show final answer
        
        # Add agent reply to history
        st.session_state.history.append(("assistant", agent_response))
        st.session_state.last_agent = handled_by









