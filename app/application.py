import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = Flask(__name__)
# Keep a stable secret key across worker recycles if set in env, or fallback to random
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        # Import only on demand to prevent memory spikes during cold boots
        from app.components.retriever import create_qa_chain
        qa_chain = create_qa_chain()
    return qa_chain

def clean_response(text: str) -> str:
    """Removes model reasoning/thinking tokens and stray HTML artifacts."""
    if not text:
        return "No response"

    # Remove standard <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Extract final answer if reasoning dump is prefixed
    if "[Output]:" in text:
        text = text.split("[Output]:")[-1]
    elif "Output generation." in text:
        text = text.split("Output generation.")[-1]

    # Clean up literal <br> strings or stray tags
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input and user_input.strip():
            user_input = user_input.strip()
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                chain = get_qa_chain()
                if chain is None:
                    raise RuntimeError("QA chain could not be created (LLM or VectorStore issue)")

                response = chain.invoke({"input": user_input})
                raw_answer = response.get("answer", "No response")
                
                # Sanitize the output
                cleaned_answer = clean_response(raw_answer)

                messages.append({"role": "assistant", "content": cleaned_answer})
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)

            return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear", methods=["GET", "POST"])
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)