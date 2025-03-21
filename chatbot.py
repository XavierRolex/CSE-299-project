import streamlit as st
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Path to the stored Chroma database
db_dir = "./chroma_db"

# Load the stored Chroma vector database with the same embedding model
embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
try:
    vector_db = Chroma(collection_name="local-rag", persist_directory=db_dir, embedding_function=embedding)
    print("Vector DB loaded successfully!")
except Exception as e:
    st.error(f"Failed to load vector database: {e}")
    raise

# Set up LLM and retrieval
local_model = "llama3.2"  # or whichever model you prefer
llm = ChatOllama(model=local_model)

# Query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant of a banking system that 
    assists the clients of that bank. Your task is to generate 2
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Set up retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt template
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
if you do not know the answer just tell to contact with the bank's helpline number
which is 16234
"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chatbot logic
def get_bot_response(user_message):
    try:
        responses = chain.invoke({"question": user_message})  # Pass input as a dictionary
        time.sleep(1)  # Simulates a delay in bot response
        return responses
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for styling
custom_css = """
<style>
    /* Chat container */
    .chat-container {
        max-width: 600px;
        margin: auto;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* User message */
    .user-message {
        text-align: right;
        margin-bottom: 10px;
    }

    .user-message div {
        display: inline-block;
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
    }

    /* Bot message */
    .bot-message {
        text-align: left;
        margin-bottom: 10px;
    }

    .bot-message div {
        display: inline-block;
        background-color: #e9ecef;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
    }

    /* Timestamp */
    .timestamp {
        font-size: 12px;
        color: #6c757d;
        margin-top: 5px;
    }

    /* Input field */
    .input-container {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }

    .input-container input {
        flex: 1;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
</style>
"""
# Apply page name
st.set_page_config(page_title="Bank Service Chatbot", page_icon="🏦", layout="centered")

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.title("🏛️ Bank Service Chatbot 🤖")

# Placeholder for chat history
chat_placeholder = st.empty()

# Display chat history
def render_chat_history():
    chat_content = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            chat_content += f'<div class="user-message"><div>{message["text"]}</div></div>'
        else:
            chat_content += f'<div class="bot-message"><div>{message["text"]}</div><div class="timestamp">Response time: {message["response_time"]} seconds</div></div>'
    chat_placeholder.markdown(chat_content, unsafe_allow_html=True)

render_chat_history()

# Input field for user message
st.markdown('<div class="input-container">', unsafe_allow_html=True)
current_input = st.text_input("", placeholder="Type your message here...", key="user_input", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# Function to handle sending messages
def send_message(input_text):
    if input_text.strip():  # Check if the input is not empty
        # Save user message
        st.session_state.chat_history.append({"role": "user", "text": input_text})

        # Start the timer
        start_time = time.time()

        # Get bot response
        bot_response = get_bot_response(input_text)

        # Calculate response time
        end_time = time.time()
        response_time = round(end_time - start_time, 2)  # Round to 2 decimal places

        # Save bot response with response time
        st.session_state.chat_history.append({
            "role": "bot",
            "text": bot_response,
            "response_time": response_time
        })

        # Re-render the chat history
        render_chat_history()

# Handle Enter key press
if current_input and st.session_state.get("last_input") != current_input:
    send_message(current_input)  # Trigger the bot response logic
    st.session_state.last_input = current_input  # Track the last processed input
    current_input = ""  # Reset the input field visually (without modifying session state)