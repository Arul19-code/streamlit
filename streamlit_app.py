import streamlit as st
from transformers import pipeline

# ---- Page config ----
st.set_page_config(page_title="Mini GPT", layout="centered")

# ---- Load model ----
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")  # chat-friendly model
generator = load_model()

# ---- UI ----
st.markdown("<h2 style='text-align:center;'>🤖 Mini GPT Chatbot</h2>", unsafe_allow_html=True)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# Input
user_input = st.text_input("Type your message:")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate reply
    with st.spinner("Bot is typing..."):
        response = generator(user_input, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)[0]["generated_text"]
        reply = response.strip()

    st.session_state.messages.append({"role": "bot", "content": reply})
    st.experimental_rerun()  # refresh to show new messages