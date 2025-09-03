import streamlit as st
from transformers import pipeline

# ---- Streamlit page config ----
st.set_page_config(page_title="KNIGHT ChatGPT", layout="centered")

# ---- Load model (cached) ----
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

# ---- UI ----
st.markdown("<h2 style='text-align:center;'>KNIGHT ChatGPT</h2>", unsafe_allow_html=True)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# Input
user_input = st.text_input("Type your message:")
if st.button("Send") and user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate reply
    with st.spinner("Bot is typing..."):
        prompt = f"You are a helpful assistant.\nUser: {user_input}\nAssistant:"
        response = generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )[0]["generated_text"]
        reply = response[len(prompt):].strip()

    # Save & display reply
    st.session_state.messages.append({"role": "bot", "content": reply})
    st.experimental_rerun()