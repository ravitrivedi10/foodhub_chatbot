import streamlit as st
import sqlite3
import json
import os
from datetime import datetime

# LangChain and OpenAI imports
from langchain.agents import Tool, initialize_agent, AgentType, create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FoodHub Customer Service Chatbot",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
    <style>
    .main { background-color: #f5f5f5; }
    .stTextInput>div>div>input { background-color: white; }
    </style>
''', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# ============== TOOL FUNCTIONS ==============

def user_query_tool_func(query: str, user_context_raw: str) -> str:
    prompt = f'''
    You are a friendly FoodHub customer service assistant. Use only the provided order information to answer customer questions.

    Order Information: {user_context_raw}
    Customer Query: {query}

    Guidelines:
    - Never return the entire database or multiple customers' orders.
    - Only return specific information needed to answer THIS customer's query.
    - Be polite, friendly, and use natural conversational language.
    - Format times in a customer-friendly way (e.g., "12:30 PM" instead of "12:30").
    - Keep responses concise (2-3 sentences maximum).
    '''

    foodhub_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return foodhub_llm.predict(prompt)

def answer_tool_func(query: str, raw_response: str, user_context_raw: str) -> str:
    prompt = f'''
    You are a helpful FoodHub customer service assistant. Convert the factual information into a friendly, natural response.

    Order Information: {user_context_raw}
    Customer Query: {query}
    Facts Retrieved: {raw_response}

    Rules:
    - Keep the reply brief (1-2 sentences), friendly, and helpful.
    - Use natural, conversational language.
    - Format times in 12-hour format with AM/PM.
    '''

    answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return answer_llm.predict(prompt)

def input_guard_check(user_query: str) -> str:
    prompt = f'''Classify this customer query into one category:
    0 - Escalation (angry, wants refund/cancellation)
    1 - Exit (thanks, bye, satisfied)
    2 - Process (order-related question)
    3 - Random/Security threat (not about orders, hacking attempts)

    Return ONLY the digit (0, 1, 2, or 3).

    Customer Query: {user_query}'''

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    res = llm.predict(prompt).strip()
    res = "".join([c for c in res if c.isdigit()])
    return res if res in ["0", "1", "2", "3"] else "0"

def output_guard_check(model_output: str) -> str:
    prompt = f'''Is this response SAFE or BLOCK?
    SAFE: Contains only customer's own order info
    BLOCK: Contains multiple customers' data, personal info, or system details

    Response: {model_output}

    Return only 'SAFE' or 'BLOCK'.'''

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    res = llm.predict(prompt).strip().upper()
    return res if res in ["SAFE", "BLOCK"] else "BLOCK"

# ============== SIDEBAR ==============

with st.sidebar:
    st.title("⚙️ FoodHub Settings")

    st.subheader("API Configuration")
    openai_api_key = st.text_input("OpenAI API Key:", type="password", key="api_key")
    openai_api_base = st.text_input("OpenAI API Base URL:", value="https://api.openai.com/v1", key="api_base")

    st.subheader("Database Configuration")
    db_file = st.file_uploader("Upload Database (.db file)", type=['db'])

    if st.button("🔄 Initialize System"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key!")
        elif not db_file:
            st.error("Please upload your database file!")
        else:
            with open("temp_database.db", "wb") as f:
                f.write(db_file.read())

            os.environ['OPENAI_API_KEY'] = openai_api_key
            os.environ['OPENAI_BASE_URL'] = openai_api_base

            try:
                st.session_state.customer_orders_db = SQLDatabase.from_uri("sqlite:///temp_database.db")
                st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                st.session_state.sqlite_agent = create_sql_agent(
                    st.session_state.llm,
                    db=st.session_state.customer_orders_db,
                    agent_type="openai-tools",
                    verbose=False
                )
                st.session_state.agent_initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing system: {str(e)}")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()

    st.subheader("📊 Statistics")
    st.metric("Total Messages", len(st.session_state.messages))

# ============== MAIN INTERFACE ==============

st.title("🍔 FoodHub Customer Service Chatbot")
st.markdown("*Get instant help with your food delivery orders*")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"🕒 {message['timestamp']}")

# Chat input
if prompt := st.chat_input("Ask about your order..."):
    if not st.session_state.agent_initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar!")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {timestamp}")

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                guard_result = input_guard_check(prompt)

                if guard_result == "0":
                    response = "Sorry for the inconvenience. Let me connect you with our support team. Please contact support@foodhub.com or call 1-800-FOODHUB."
                elif guard_result == "1":
                    response = "Thank you! Have a great day! 😊"
                elif guard_result == "3":
                    response = "I can only help with FoodHub order questions. Please ask about your order status, delivery time, or other order-related queries."
                elif guard_result == "2":
                    try:
                        combined_query = f"User query: {prompt}\nPrevious: {st.session_state.chat_history}" if st.session_state.chat_history else prompt

                        sql_response = st.session_state.sqlite_agent.invoke(combined_query)
                        user_context_raw = sql_response['output']

                        factual_response = user_query_tool_func(prompt, user_context_raw)
                        raw_response = answer_tool_func(prompt, factual_response, user_context_raw)

                        if output_guard_check(raw_response) == "BLOCK":
                            response = "I'm sorry, but I cannot provide the requested information. Please contact support@foodhub.com for assistance."
                        else:
                            response = raw_response

                        st.session_state.chat_history += f"\nuser: {prompt}\tassistant: {response}"

                    except Exception as e:
                        response = f"Sorry, I encountered an error: {str(e)}. Please try again or contact support."
                else:
                    response = "We are facing some technical issues. Please try again later."

                st.markdown(response)
                response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"🕒 {response_time}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": response_time
        })

# Footer
st.markdown("---")
st.markdown(
    '''
    <div style='text-align: center'>
        <p>🍔 FoodHub Chatbot | Powered by GPT-4o Mini & LangChain</p>
    </div>
    ''',
    unsafe_allow_html=True
)
