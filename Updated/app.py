import streamlit as st
from ui import render_sidebar, render_chat_interface

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Bank Chatbot", page_icon="🏦", layout="centered")
render_sidebar()
render_chat_interface()
