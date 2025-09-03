import streamlit as st
from transformers import pipeline

# ---- Page config ----
st.set_page_config(page_title="Mini GPT", layout="centered")

# ---- Load model ----
@st.cache_resource
def load_model():
    # Chat-friendly small model
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

generator = load_model()

# ---- UI ----
st.markdown("<h2 style='text-align:center;'>🤖 KNIGHT ChatGPT</h2>", unsafe_allow_html=True)

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display chat history ----
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(f"<div style='background:#0b93f6;color:white;padding:10px;border-radius:12px;margin:5px 0;max-width:70%;margin-left:auto;text-align:right'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#444654;color:#ececec;padding:10px;border-radius:12px;margin:5px 0;max-width:70%;margin-right:auto;text-align:left'>{content}</div>", unsafe_allow_html=True)

# ---- Input form ----
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ---- Generate bot reply ----
    with st.spinner("🤖 Bot is typing..."):
        response = generator(
            user_input,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]

        # Clean reply
        reply = response.strip()
        st.session_state.messages.append({"role": "bot", "content": reply})
