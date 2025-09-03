import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---- Load model ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="Mini GPT", layout="centered")
st.markdown("<h2 style='text-align:center;'>🤖 Mini GPT Chatbot</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# Input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Bot is typing..."):
        prompt = f"You are a helpful assistant.\nUser: {user_input}\nAssistant:"
        response = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
        reply = response[len(prompt):].strip()

    st.session_state.messages.append({"role": "bot", "content": reply})
    st.markdown(f"**🤖 Bot:** {reply}")