# ====== Imports ======
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, MessageGraph
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import openai
import numpy as np
import pandas as pd
import re
import streamlit as st  
import duckdb
from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# ====== CONFIGURATION ======
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI client for Response API
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Keep existing LLM for other agents
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ====== DuckDB / Parquet paths ======
DUCKDB_PATH = "orders.duckdb"
ORDERS_PARQUET = "orders.parquet"

# Pinecone settings
INDEX_NAME = "instruments"
DIMENSION = 1536
NAMESPACE = "default_namespace"

# ====== PERFORMANCE OPTIMIZATION: Global Caches ======
_customer_cache = {}  # Cache for customer purchase history
_recommendation_agents = {}  # Cache for recommendation agents

# ====== DATABASE CONNECTION (OPTIMIZED) ======
_DUCK_CONN = None
def get_duckdb_con():
    """Create (once) and return a DuckDB connection with a view `orders` over the parquet."""
    global _DUCK_CONN
    if _DUCK_CONN is None:
        _DUCK_CONN = duckdb.connect(DUCKDB_PATH, read_only=False)
        p = ORDERS_PARQUET.replace("'", "''")  # escape just in case
        _DUCK_CONN.execute(f"CREATE OR REPLACE VIEW orders AS SELECT * FROM read_parquet('{p}')")
        _DUCK_CONN.execute("PRAGMA enable_object_cache;")
        # Performance optimizations for DuckDB
        _DUCK_CONN.execute("PRAGMA memory_limit='2GB';")
        _DUCK_CONN.execute("PRAGMA threads=4;")
    return _DUCK_CONN

# Initialize connection at startup
_ = get_duckdb_con()

# ====== PINECONE SETUP ======
pc = Pinecone(api_key=PINECONE_API_KEY)
existing = [ix["name"] for ix in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

# Embeddings + VectorStore
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=emb_model,
    namespace=NAMESPACE,
)

# ====== HELPER FUNCTIONS FOR RECOMMENDATION AGENT ======
def parse_list_field(value: Any) -> List[str]:
    """Helper to parse JSON-like strings, lists, or NumPy arrays into a list of strings."""
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value.flatten() if v]

    if isinstance(value, list):
        return [str(v).strip() for v in value if v]

    if not isinstance(value, str):
        return []

    value = value.strip().strip('[]').strip()
    if not value:
        return []

    try:
        import json
        parsed = json.loads(value if value.startswith('[') else f'[{value}]')
        return [str(p).strip() for p in parsed if p]
    except json.JSONDecodeError:
        items = []
        current_item = ""
        in_quotes = False
        quote_char = None
        i = 0
        while i < len(value):
            char = value[i]
            if char in ('"', "'") and (i == 0 or value[i-1] != '\\'):
                if in_quotes and char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    in_quotes = True
                    quote_char = char
                current_item += char
            elif char == ',' and not in_quotes:
                items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char
            i += 1
        if current_item:
            items.append(current_item.strip())

        cleaned_items = [item.strip(" '\"") for item in items if item]
        return cleaned_items

    return []

def get_purchase_history(customer_id: str, limit_rows: int = 200):
    """Fetch ONLY products_ordered and items_type for this customer with caching."""
    if not customer_id:
        return {"customer_id": "", "recent_products": [], "recent_types": [], "is_new_customer": True}

    # OPTIMIZATION 1: Check cache first
    cache_key = f"purchase_history_{customer_id}"
    if cache_key in _customer_cache:
        # print(f"✅ Using cached purchase history for {customer_id}")
        return _customer_cache[cache_key]

    con = get_duckdb_con()
    try:
        df = con.execute(
            """
            SELECT products_ordered, items_type
            FROM orders
            WHERE customer_id = ?
            ORDER BY order_purchase_timestamp DESC
            LIMIT ?
            """,
            [customer_id, limit_rows]
        ).fetchdf()

        if df.empty:
            result = {
                "customer_id": customer_id,
                "recent_products": [],
                "recent_types": [],
                "is_new_customer": True
            }
        else:
            prod_lists = [parse_list_field(v) for v in df["products_ordered"].tolist()]
            type_lists = [parse_list_field(v) for v in df["items_type"].tolist()]

            recent_products = [p for sub in prod_lists for p in sub]
            recent_types = [t for sub in type_lists for t in sub]

            result = {
                "customer_id": customer_id,
                "recent_products": recent_products,
                "recent_types": recent_types,
                "is_new_customer": False
            }

        # OPTIMIZATION 1: Cache the result
        _customer_cache[cache_key] = result
        print(f"✅ Cached purchase history for {customer_id}")
        return result

    except Exception as e:
        print(f"[get_purchase_history] error: {e}")
        result = {
            "customer_id": customer_id,
            "recent_products": [],
            "recent_types": [],
            "is_new_customer": True
        }
        # Cache even error results to avoid repeated failures
        _customer_cache[cache_key] = result
        return result

def catalog_search_tool(query: str, k: int = 2) -> Dict[str, Any]:
    """Search ONLY the Pinecone catalog. Returns raw page_content for products."""
    docs = vectorstore.similarity_search(query, k=k)
    return {"documents": [d.page_content for d in docs]}

def clean_purchase_history(recent_products: List[Any]) -> List[str]:
    """Clean the recent_products list using your working logic"""
    owned_queries: List[str] = []

    for item in recent_products:
        if isinstance(item, str) and item.strip().startswith("["):
            s = item.strip()
            try:
                s_cleaned = s.strip(" '\"[]")
                parts = [p.strip(" '\"") for p in s_cleaned.split('\n') if p.strip(" '\"")]
                owned_queries.extend(parts)
            except Exception:
                quoted = re.findall(r"""(['"])(.*?)\1""", s, flags=re.DOTALL)
                if quoted:
                    owned_queries.extend([q[1].replace("\n", " ").strip() for q in quoted])
                else:
                    parts = [p.strip(" '\"\t[]") for p in re.split(r"[\n,]+", s)]
                    owned_queries.extend([p for p in parts if p])
        else:
            owned_queries.append(item)

    owned_queries = [
        oq.strip(" '\"\n[]")
        for oq in owned_queries
        if isinstance(oq, str) and oq.strip()
    ]
    return list(dict.fromkeys(owned_queries))

def get_owned_profile(recent_products: List[Any]) -> List[str]:
    """Get exact owned product documents from catalog using optimized batch search"""
    owned_queries = clean_purchase_history(recent_products)
    
    if not owned_queries:
        return []
    
    owned_docs = []
    
    # OPTIMIZATION 3: Batch vector searches instead of individual searches
    print(f"🔍 Searching for {len(owned_queries)} owned products...")
    
    # Search for multiple products at once with higher k value
    for query_batch in [owned_queries[i:i+5] for i in range(0, len(owned_queries), 5)]:
        batch_query = " OR ".join(query_batch)
        result = catalog_search_tool(batch_query, k=min(len(query_batch) * 2, 10))
        docs = result.get("documents", []) or []
        
        for oq in query_batch:
            for d in docs:
                if not isinstance(d, str):
                    continue
                pname = ""
                for line in d.splitlines():
                    if line.startswith("Product Name:"):
                        pname = line.replace("Product Name:", "").strip()
                        break
                if pname == oq:
                    print("Analysing the detailed features of --:", pname)
                    owned_docs.append(d)
                    break

    return owned_docs

def fetch_customer_purchase_history_with_docs(customer_id: str) -> Dict[str, Any]:
    """Fetch customer purchase history and associated product documents with caching."""
    # OPTIMIZATION 1: Check if full customer data is already cached
    full_cache_key = f"customer_full_{customer_id}"
    if full_cache_key in _customer_cache:
        # print(f"✅ Using cached full customer data for {customer_id}")
        return _customer_cache[full_cache_key]

    purchase_history = get_purchase_history(customer_id)

    purchased_product_docs = []
    if not purchase_history["is_new_customer"]:
        purchased_product_docs = get_owned_profile(purchase_history["recent_products"])

    result = {
        "customer_id": purchase_history["customer_id"],
        "recent_products": purchase_history["recent_products"],
        "recent_types": purchase_history["recent_types"],
        "is_new_customer": purchase_history["is_new_customer"],
        "purchased_product_docs": purchased_product_docs
    }

    # OPTIMIZATION 1: Cache the full result
    _customer_cache[full_cache_key] = result
    # print(f"✅ Cached full customer data for {customer_id}")
    
    return result

# ====== MUSIC STORE AGENT CLASS ======
@dataclass
class AgentConfig:
    openai_api_key: str
    model: str = "gpt-4.1-mini"

class MusicStoreAgent:
    def __init__(self, config: AgentConfig):
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.model = config.model
        self.conversation_id = None
        self.customer_context = None
        
        self.tools = [
            {
                "type": "function",
                "name": "catalog_search_tool",
                "description": "Search the music store catalog for products",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for music products"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 2
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
    def initialize_customer_session(self, customer_id: str) -> Dict[str, Any]:
        """Initialize a new session for a customer by loading their data and creating conversation"""
        print(f"Loading customer data for: {customer_id}")
        
        # OPTIMIZATION 1: Use cached customer data fetch
        customer_data = fetch_customer_purchase_history_with_docs(customer_id)
        self.customer_context = customer_data
        
        system_message = self._create_system_message(customer_data)
        
        conversation_response = self.client.conversations.create(
            metadata={"customer_id": customer_id}
        )
        
        self.client.conversations.items.create(
            conversation_id=conversation_response.id,
            items=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_message}]
                }
            ]
        )
        
        self.conversation_id = conversation_response.id
        
        print(f"✅ Session initialized for customer {customer_id}")
        print(f"🛒 Customer type: {'New Customer' if customer_data['is_new_customer'] else 'Returning Customer'}")
        print(f"📦 Owned products: {len(customer_data['recent_products'])}")
        
        return {
            "conversation_id": self.conversation_id,
            "customer_data": customer_data,
            "status": "initialized"
        }
    
    def _create_system_message(self, customer_data: Dict[str, Any]) -> str:
        """Create a comprehensive system message with customer context"""
        if customer_data['is_new_customer']:
            system_message = """You are an AI assistant helping a human customer service agent provide music store recommendations during a live call. 

CUSTOMER STATUS: NEW CUSTOMER
- This customer has no purchase history
- Provide general product education and broad recommendations
- Ask about their musical interests and experience level
- Suggest starter products and beginner-friendly options

TOOL USAGE RULES:
- ONLY use catalog_search_tool when the customer specifically asks about products, wants recommendations, or mentions specific instrument types
- DO NOT use tools for greetings, general conversation, or non-product related questions
- Use tools only when you need to search for actual products to help the customer
- ALWAYS get ONLY two relevant product documents when using the tool
- After receiving tool results, IMMEDIATELY provide recommendations based on those results
- Do NOT make additional tool calls once you have received product data
- Use the search results to give specific product recommendations

CAPABILITIES:
- Search the music catalog using catalog_search_tool ONLY when product search is needed
- Provide detailed product comparisons and explanations
- Help customers understand different product categories (Electric guitar, electric drum, Keyboards)
- Make personalized recommendations

RESPONSE FORMAT:
- Frame responses as guidance to the human agent, not direct customer responses
- Give specific talking points about why recommended products are best for them
- Use phrases like "You could ask...", "I'd recommend suggesting...", "Based on this, you might want to..."

Example: Instead of "What type of music do you play?", say "Ask them what type of music they're interested in playing to better understand their needs."

IMPORTANT RULES:
- Always be helpful and knowledgeable about music products
- Ask clarifying questions to understand customer needs
- Suggest products that match their skill level and interests
- NEVER tell the human agent to use any tools, catalog_search_tool or any internal tools in your response
- NEVER tell human agent to search for products by themselve"""

        else:
            owned_products_str = ", ".join(customer_data['recent_products'][:10])
            owned_types_str = ", ".join(set(customer_data['recent_types']))
            
            product_docs_summary = "\n".join([
                f"OWNED PRODUCT {i+1}:\n{doc[:300]}..." 
                for i, doc in enumerate(customer_data['purchased_product_docs'][:5])
            ])
            
            system_message = f"""You are an AI assistant helping a human customer service agent serve a returning customer during a live call.


CUSTOMER STATUS: RETURNING CUSTOMER

OWNED PRODUCTS: {owned_products_str}
PRODUCT CATEGORIES OWNED: {owned_types_str}

DETAILED PRODUCT INFORMATION:
{product_docs_summary}

TOOL USAGE RULES:
- After you receive catalog search results, use them to answer the user's question. Do not call the tool again for the same query. Respond to the user with a recommendation.
- Do NOT use tools for explaining previous recommendations
- Do NOT use tools when asked 'why' about already shown products
- ONLY use catalog_search_tool when the customer specifically asks about products, wants recommendations, or mentions specific instrument types
- DO NOT use tools for greetings, general conversation, or non-product related questions
- For "hello", "hi", general questions, respond normally without any tool calls
- Use tools only when you need to search for actual products to help the customer


CAPABILITIES:
- Search the music catalog using catalog_search_tool ONLY when product search is needed
- Compare new products with what the customer already owns
- Suggest upgrades, complementary products, or accessories
- Provide detailed explanations about why certain products are recommended
- ALWAYS get two relevant product documents when using the tool

RESPONSE FORMAT:
- Provide guidance to the human agent, not direct customer responses
- Use "Tell them...", "You should mention...", "I'd suggest recommending..."
- Give specific talking points about why certain products complement their existing gear
- Alert about products to avoid (already owned)

Example: Instead of "I see you have a Fender guitar", say "Remind them they already have a Fender guitar, so you might suggest a complementary amplifier or effects pedal instead."

IMPORTANT RULES:
- NEVER recommend products the customer already owns (check the owned products list above)
- Focus on complementary products, upgrades, or different categories
- ALWAYS Reference their existing products when making recommendations
- Explain how new products work with their current setup
- If they ask about a product they already own, acknowledge it and suggest alternatives or accessories
- NEVER tell the human agent to use any tools, catalog_search_tool or any internal tools in your response
- NEVER tell human agent to search for products by themselve"""


        return system_message
    
    def chat(self, user_message: str) -> str:
        """Handle user chat message and return agent response"""
        if not self.conversation_id:
            return "❌ Please initialize customer session first using initialize_customer_session()"
        
        try:
            self.client.conversations.items.create(
                conversation_id=self.conversation_id,
                items=[
                    {
                        "type": "message", 
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_message}]
                    }
                ]
            )
            
            # Step 1: Initial LLM response (can use tools)
            response = self.client.responses.create(
                model=self.model,
                tools=self.tools,
                conversation=self.conversation_id,
                parallel_tool_calls=False,
                input=[{"role": "user", "content": user_message}]
            )

            # Step 2: Check if tool calls were made
            tool_calls = []
            if hasattr(response, 'output') and response.output:
                for out in response.output:
                    if hasattr(out, 'type') and out.type == "function_call":
                        tool_calls.append(out)

            # Step 3: If no tool calls, extract text response
            if not tool_calls:
                assistant_content = ""
                if hasattr(response, 'output') and response.output:
                    for out in response.output:
                        if hasattr(out, 'content'):
                            if isinstance(out.content, str):
                                assistant_content += out.content
                            elif hasattr(out.content, 'text'):
                                assistant_content += out.content.text
                            elif isinstance(out.content, list) and len(out.content) > 0:
                                if hasattr(out.content[0], 'text'):
                                    assistant_content += out.content[0].text
                                elif isinstance(out.content[0], str):
                                    assistant_content += out.content[0]
                return assistant_content if assistant_content else "No response generated."

            # Step 4: Execute tool calls
            tool_outputs = []
            for tool_call in tool_calls:
                tool_name = getattr(tool_call, 'name', None)
                tool_arguments = getattr(tool_call, 'arguments', None)
                call_id = getattr(tool_call, 'call_id', f"call_{len(tool_outputs)}")

                if tool_name == "catalog_search_tool":
                    if isinstance(tool_arguments, str):
                        args = json.loads(tool_arguments)
                    else:
                        args = tool_arguments

                    result = self.handle_tool_call("catalog_search_tool", args)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result)
                    })

            # Step 5: FORCE text response with tools DISABLED
            # final_response = self.client.responses.create(
            #     model=self.model,
            #     conversation=self.conversation_id,
            #     tools=[],  # Empty tools
            #     input=tool_outputs + [{
            #             "role": "system", 
            #             "content": "Based on the tool results above, provide a final text response. NEVER call any more tools."
            #         }]
            # )
            final_response = self.client.responses.create(
                model=self.model,
                conversation=self.conversation_id,
                tools=[],  # Disable further tool calls
                input=[
                    {
                        "role": "system",
                        "content": "You have received tool results. Summarize them into a clear and helpful final response. NEVER call more tools."
                    },
                    {
                        "role": "user",
                        "content": user_message  # the original query
                    },
                    *tool_outputs
                ]
            )


            # Step 6: Extract text from final response
            assistant_content = ""
            if hasattr(final_response, 'output') and final_response.output:
                for out in final_response.output:
                    if hasattr(out, 'content'):
                        if isinstance(out.content, str):
                            assistant_content += out.content
                        elif hasattr(out.content, 'text'):
                            assistant_content += out.content.text
                        elif isinstance(out.content, list) and len(out.content) > 0:
                            if hasattr(out.content[0], 'text'):
                                assistant_content += out.content[0].text
                            elif isinstance(out.content[0], str):
                                assistant_content += out.content[0]

            return assistant_content if assistant_content else "No response generated."
            
        except Exception as e:
            return f"❌ Error processing your message: {str(e)}"

    def handle_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls from the agent (catalog search)"""
        if tool_name == "catalog_search_tool":
            query = tool_args.get("query", "")
            k = tool_args.get("k", 2)
            
            print(f"🔍 Searching catalog to get relevant instruments")
            result = catalog_search_tool(query, k)
            
            return result
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def end_session(self):
        """Clean up the session"""
        print(f"🔚 Ending session. Conversation ID: {self.conversation_id}")
        self.conversation_id = None
        self.customer_context = None

# ====== Graph State ======
class AgentState(TypedDict):
    query: str
    customer_id: str
    response: str
    next_node: str
    last_agent: str
    history: List[str]
    router_conversation_id: str

# ====== Router Agent with Response API ======
def initialize_router_conversation():
    """Initialize a new conversation for routing decisions"""
    try:
        conversation_response = openai_client.conversations.create(
            metadata={"purpose": "routing_decisions"}
        )
        
        system_message = """You are a routing classifier for a customer support assistant.

ROUTE the user's latest message to EXACTLY one of:
- order_tracking_agent → order status, shipping, delivery, ETA, delays, cancel/modify order, or explicit order/customer IDs.
- recommendation_agent → product recommendations, buying intent, product comparisons, "I want to buy", "looking for", "need a guitar", follow-up questions about recommended products, "why is this the best", product features, shopping assistance.
- general_agent → greetings/small talk/thanks, company policies (returns/shipping policy), store hours, careers, basic product information requests.

DECISION CHECKLIST (think silently; output JSON only):
1) CONTEXT AWARENESS: If the conversation shows previous product recommendations or shopping discussion and the new message relates to those products (asks "why", "tell me more", "which is better", "what about features"), route to recommendation_agent.

2) BUYING INTENT SIGNALS:
   • "I want to buy", "looking for", "need a", "searching for", "show me" + product types → recommendation_agent
   • Product comparisons, "which is better", "recommend me" → recommendation_agent
   • Questions about product features, pricing, availability → recommendation_agent

3) ORDER TRACKING SIGNALS:
   • Order ID like ORD\\d{6} or customer ID like CUST\\d{4,} → order_tracking_agent
   • Shipping terms (track, delivery, shipped, arrive, ETA, delay, carrier) → order_tracking_agent

4) FOLLOW-UP AWARENESS:
   • If last_agent was recommendation_agent and query relates to products/shopping → recommendation_agent
   • If last_agent was order_tracking_agent and query about order status/timing → order_tracking_agent

5) GENERAL QUERIES:
   • Greetings, thanks, policies, store hours, careers → general_agent
   • Non-specific product questions without buying intent → general_agent

OUTPUT FORMAT:
Return ONLY a JSON object like: {"next_node":"recommendation_agent"}
Allowed values: "order_tracking_agent", "recommendation_agent", "general_agent".
No explanations. No extra fields."""
        
        openai_client.conversations.items.create(
            conversation_id=conversation_response.id,
            items=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_message}]
                }
            ]
        )
        
        return conversation_response.id
    except Exception as e:
        print(f"Error initializing router conversation: {e}")
        return None

def router_agent(state: AgentState) -> AgentState:
    print("Router Agent Thinking........")
    q = state["query"]
    router_conv_id = state.get("router_conversation_id", "")

    # Agar pehli dafa hai to conversation create karo
    if not router_conv_id:
        router_conv_id = initialize_router_conversation()
        if not router_conv_id:
            return {"next_node": "general_agent", "router_conversation_id": ""}

    # User message ko conversation mein add karo
    openai_client.conversations.items.create(
        conversation_id=router_conv_id,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": q}]
            }
        ]
    )

    # LLM se routing decision lo
    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        conversation=router_conv_id,
        input=[{"role": "user", "content": q}]
    )

    assistant_content = ""
    if hasattr(response, 'output') and response.output:
        for out in response.output:
            if hasattr(out, 'content'):
                if isinstance(out.content, str):
                    assistant_content += out.content
                elif hasattr(out.content, 'text'):
                    assistant_content += out.content.text
                elif isinstance(out.content, list) and len(out.content) > 0:
                    if hasattr(out.content[0], 'text'):
                        assistant_content += out.content[0].text
                    elif isinstance(out.content[0], str):
                        assistant_content += out.content[0]

    # Parse LLM output for next_node
    next_node = "general_agent"
    try:
        obj = json.loads(assistant_content.strip())
        cand = obj.get("next_node", "")
        if cand in ("order_tracking_agent", "recommendation_agent", "general_agent"):
            next_node = cand
    except Exception:
        pass

    print(f"Router decision: {next_node}")
    return {"next_node": next_node, "router_conversation_id": router_conv_id}

# ====== Recommendation Agent (OPTIMIZED) ======
def recommendation_agent(state: AgentState) -> AgentState:
    """Recommendation agent using MusicStoreAgent with caching"""
    print("Recommendation Agent thinking.......")
    
    customer_id = state["customer_id"]
    user_query = state["query"]
    
    try:
        # LAZY INITIALIZATION: Check if agent exists, create if not
        if customer_id not in _recommendation_agents:
            print(f"🔧 Initializing new recommendation agent for customer: {customer_id}")
            agent = MusicStoreAgent(AgentConfig(openai_api_key=OPENAI_API_KEY))
            agent.initialize_customer_session(customer_id)
            _recommendation_agents[customer_id] = agent
            print(f"✅ Agent initialized for customer: {customer_id}")
        
        agent = _recommendation_agents[customer_id]
        response = agent.chat(user_query)
        
        return {
            "response": response,
            "last_agent": "recommendation_agent"
        }
        
    except Exception as e:
        print(f"Error in recommendation agent: {e}")
        return {
            "response": f"Sorry, I encountered an error processing your request: {str(e)}. Please try again.",
            "last_agent": "recommendation_agent"
        }

def order_tracking_agent(state: AgentState) -> AgentState:
    """Order tracking via DuckDB + LLM."""
    print("Order Tracking Agent Thinking.........")
    user_text = state["query"]
    customer_id_from_state = state.get("customer_id", "")

    m_order = re.search(r"\bORD\d{6}\b", user_text, flags=re.IGNORECASE)
    m_cust  = re.search(r"\bCUST\d{4,}\b", user_text, flags=re.IGNORECASE)
    extracted_order_id = m_order.group(0).upper() if m_order else ""
    extracted_customer_id = (m_cust.group(0).upper() if m_cust else customer_id_from_state)

    required_cols = """
        order_id,
        order_status,
        CAST(order_purchase_timestamp AS TIMESTAMP)      AS order_purchase_timestamp,
        CAST(order_estimated_delivery_date AS TIMESTAMP) AS order_estimated_delivery_date,
        CAST(order_delivered_customer_date AS TIMESTAMP) AS order_delivered_customer_date,
        products_ordered,
        items_type
    """

    con = get_duckdb_con()
    params = []
    total_count = None

    try:
        if extracted_order_id:
            sql = f"SELECT {required_cols} FROM orders WHERE order_id = ? LIMIT 1"
            params = [extracted_order_id]
            df = con.execute(sql, params).fetchdf()

        elif extracted_customer_id:
            sql = f"""
                SELECT {required_cols}
                FROM orders
                WHERE customer_id = ?
                ORDER BY order_purchase_timestamp DESC
                LIMIT 5
            """
            params = [extracted_customer_id]
            df = con.execute(sql, params).fetchdf()

            count_sql = "SELECT COUNT(*) FROM orders WHERE customer_id = ?"
            total_count = con.execute(count_sql, [extracted_customer_id]).fetchone()[0]

        else:
            return {"response": "Please share your customer ID like `CUST4240` or an order ID like `ORD000123`."}

        if df.empty:
            return {"response": "No ordered products"}

        def to_jsonable(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): return None
            if isinstance(x, pd.Timestamp): return x.isoformat()
            if isinstance(x, (np.generic,)): return x.item()
            if isinstance(x, np.ndarray): return [to_jsonable(v) for v in x.tolist()]
            if isinstance(x, (list, tuple, pd.Series)): return [to_jsonable(v) for v in list(x)]
            return x

        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].astype(str)
        df = df.map(to_jsonable)
        rows = df.to_dict(orient="records")

        meta = None
        if extracted_customer_id:
            meta = {
                "customer_id": extracted_customer_id,
                "returned": len(rows),
                "total": int(total_count) if total_count is not None else len(rows)
            }

    except Exception as e:
        return {"response": f"Error fetching order data: {e}"}

   

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
        "You are an AI order tracking assistant helping a human customer service agent during a live call. "
        "You receive one or more order rows (status, purchase date, estimated delivery, delivered date, items, categories). "
        "Your job is to extract and present only the details relevant to the user's query. "
    
        "\n\nImportant Guidelines:\n"
        "- Always phrase the response for the agent, not the customer directly.\n"
        "- If the query is about products/items → return only the list of ordered products.\n"
        "- If the query is about order status/delivery → return order status, dates, and delivery info.\n"
        "- If the query is about both → include both status + items.\n"
        "- If multiple orders are provided, summarize each one clearly.\n"
        "- Format output with short headings + bullet points so it's scannable.\n"),
        
        ("human",
        "User query: {query}\n\nOrder data:\n{order_data}\n\nMeta (optional):\n{meta}")
    ])



    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({
        "query": user_text,
        "order_data": json.dumps(rows, indent=2),
        "meta": json.dumps(meta, indent=2) if meta is not None else "null"
    })

    return {
        "response": response,
        "last_agent": "order_tracking_agent",
        "intent_lock": "tracking",
        "lock_ttl": 3,
    }

# ====== General Agent ======
def general_agent(state: AgentState) -> AgentState:
    print(" General Agent Thinking.......")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer service assistant. Handle general inquiries, company policies, greetings, and basic product information. For specific product recommendations or detailed shopping assistance, politely suggest they ask for product recommendations."),
        ("human", "User query: {query}"),
    ])
    general_chain = prompt_template | llm | StrOutputParser()
    response = general_chain.invoke({"query": state['query']})
    return {
        "response": response,
        "last_agent": "general_agent",
        "intent_lock": "",
        "lock_ttl": 0,
    }

# ====== Edge routing ======
def route_next_node(state: AgentState) -> str:
    return state['next_node']

# ====== Build graph ======
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router_agent", router_agent)
    graph.add_node("order_tracking_agent", order_tracking_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("general_agent", general_agent)
    
    graph.set_entry_point("router_agent")
    
    graph.add_conditional_edges(
        "router_agent",
        route_next_node,
        {
            "order_tracking_agent": "order_tracking_agent",
            "recommendation_agent": "recommendation_agent",
            "general_agent": "general_agent",
        }
    )
    
    graph.add_edge("order_tracking_agent", END)
    graph.add_edge("recommendation_agent", END)
    graph.add_edge("general_agent", END)
    
    return graph.compile()

# ====== Cache Management Functions ======
def clear_customer_cache(customer_id: str = None):
    """Clear cache for specific customer or all customers"""
    global _customer_cache
    if customer_id:
        keys_to_remove = [k for k in _customer_cache.keys() if customer_id in k]
        for key in keys_to_remove:
            del _customer_cache[key]
        print(f"✅ Cleared cache for customer: {customer_id}")
    else:
        _customer_cache.clear()
        print("✅ Cleared all customer cache")

def get_cache_stats():
    """Get cache statistics"""
    return {
        "customer_cache_size": len(_customer_cache),
        "recommendation_agents": len(_recommendation_agents),
        "cached_customers": list(set([k.split('_')[-1] for k in _customer_cache.keys() if 'CUST' in k]))
    }

# ====== Cleanup function (OPTIMIZED) ======
def cleanup_recommendation_agents(customer_id: str = None):
    """Clean up recommendation agents when session ends"""
    global _recommendation_agents
    if customer_id:
        if customer_id in _recommendation_agents:
            try:
                _recommendation_agents[customer_id].end_session()
                del _recommendation_agents[customer_id]
                print(f"✅ Cleaned up recommendation agent for: {customer_id}")
            except:
                pass
        # Also clear customer cache
        clear_customer_cache(customer_id)
    else:
        for customer_id, agent in _recommendation_agents.items():
            try:
                agent.end_session()
            except:
                pass
        _recommendation_agents.clear()
        _customer_cache.clear()
        print("✅ Cleaned up all recommendation agents and cache")

# ====== Main loop ======
if __name__ == "__main__":
    app = build_graph()
    customer_id = "CUST4847"

    # Pre-initialize agent and cache
    if customer_id not in _recommendation_agents:
        agent = MusicStoreAgent(AgentConfig(openai_api_key=OPENAI_API_KEY))
        agent.initialize_customer_session(customer_id)
        _recommendation_agents[customer_id] = agent
        print(f"Pre-initialized recommendation agent for customer: {customer_id}")

    history: list[str] = []
    last_agent = ""
    router_conversation_id = ""

    print("--- Starting a chat session (type 'exit' or 'quit' to end) ---")
    print(f"🚀 Performance optimizations enabled:")
    print(f"   ✅ Customer data caching")
    print(f"   ✅ DuckDB connection reuse") 
    print(f"   ✅ Vector search batching")
    print(f"   ✅ Agent instance reuse")
    print()

    try:
        while True:
            user_query = input(f"You (Customer {customer_id}): ")
            if user_query.lower() in ["exit", "quit"]:
                print("Chat session ended.")
                break
            
            # Show cache stats periodically
            if user_query.lower() == "cache":
                stats = get_cache_stats()
                print(f"📊 Cache Stats: {stats}")
                continue

            inputs = {
                "query": user_query,
                "customer_id": customer_id,
                "history": history,
                "last_agent": last_agent,
                "router_conversation_id": router_conversation_id,
            }

            final_state = app.invoke(inputs)
            agent_response = final_state["response"]
            handled_by = final_state.get("last_agent", "general_agent")
            router_conversation_id = final_state.get("router_conversation_id", "")

            print(f"Agent: {agent_response}")

            history.append(f"User: {user_query}")
            short_name = (
                "tracking" if handled_by == "order_tracking_agent"
                else "recommendation" if handled_by == "recommendation_agent"
                else "general"
            )
            history.append(f"Agent({short_name}): {agent_response}")

            if len(history) > 20:
                history = history[-20:]

            last_agent = handled_by
            
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    finally:
        cleanup_recommendation_agents()
        print("Cleanup completed.")
        print(f"📊 Final Cache Stats: {get_cache_stats()}")
