import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---- Load model (cached for speed) ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    # Use text-generation pipeline
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="Mini GPT", layout="centered")
st.markdown("<h2 style='text-align:center;'>🤖 Mini GPT Chatbot</h2>", unsafe_allow_html=True)

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# Input box
user_input = st.chat_input("Type your message...")
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"**🧑 You:** {user_input}")

    # ---- Generate bot reply ----
    with st.spinner("Bot is typing..."):
        # Instruction prompt for phi-2
        prompt = f"You are a helpful coding assistant.\nUser: {user_input}\nAssistant:"
        
        # Generate response
        response = generator(
            prompt,
            max_length=250,        # longer responses
            num_return_sequences=1,
            do_sample=True,        # adds variety
            temperature=0.7        # creativity
        )[0]["generated_text"]

        # Extract reply (remove the prompt from generated text)
        reply = response[len(prompt):].strip()

    # Save & display bot reply
    st.session_state.messages.append({"role": "bot", "content": reply})
    st.markdown(f"**🤖 Bot:** {reply}")