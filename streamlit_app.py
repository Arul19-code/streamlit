import streamlit as st
from transformers import pipeline

# Load a small Hugging Face model (runs locally, no API calls)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/phi-2")

generator = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="Mini GPT", layout="centered")

st.markdown("<h2 style='text-align:center;'>🤖 Mini GPT Chatbot</h2>", unsafe_allow_html=True)

# Session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"**🧑 You:** {user_input}")

    # Generate reply
    with st.spinner("Bot is typing..."):
        response = generator(
            user_input,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=50256  # avoids warning for GPT-2
        )[0]["generated_text"]

        reply = response[len(user_input):].strip()

    # Save & display reply
    st.session_state.messages.append({"role": "bot", "content": reply})
    st.markdown(f"**🤖 Bot:** {reply}")