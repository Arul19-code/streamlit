import streamlit as st
from transformers import pipeline

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("chat.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").lower()

    # Simple rule-based replies
    if "hello" in user_message:
        reply = "Hi there! How can I help you today?"
    elif "your name" in user_message:
        reply = "I’m Mini-GPT, your local chatbot 😊"
    elif "bye" in user_message:
        reply = "Goodbye! Have a great day!"
    else:
        reply = "Hmm... I don’t understand that yet. 🤔"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)