# ui.py
import streamlit as st
import base64
from utils import export_chat_to_pdf
from chatbot import handle_user_input


def render_sidebar():
    with st.sidebar:
        st.title("💬 Chat Options")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
        if st.button("📄 Export Chat to PDF"):
            pdf_data = export_chat_to_pdf(st.session_state.chat_history)
            b64 = base64.b64encode(pdf_data).decode("utf-8")
            href = f'<a href="data:application/pdf;base64,{b64}" download="chat_history.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)


def render_chat_interface():
    st.title("🏦 Bank Service Chatbot")

    for msg in st.session_state.chat_history:
        role = msg["role"]
        text = msg["text"]
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", key="text_input_widget", placeholder="Ask me anything...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.spinner("Thinking..."):
            response = handle_user_input(user_input)
        st.session_state.chat_history.append({"role": "bot", "text": response})
