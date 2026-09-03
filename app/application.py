import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for

load_dotenv()

# Explicitly resolve absolute path to the templates folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        # Load heavy components lazily only when the first query arrives
        from app.components.retriever import create_qa_chain
        qa_chain = create_qa_chain()
    return qa_chain

def clean_response(text: str) -> str:
    if not text:
        return "No response"
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "[Output]:" in text:
        text = text.split("[Output]:")[-1]
    elif "Output generation." in text:
        text = text.split("Output generation.")[-1]
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return text.strip()

# Health check route for Render port scanner
@app.route("/healthz", methods=["GET"])
def health():
    return "OK", 200

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
                    raise RuntimeError("QA chain could not be created.")

                response = chain.invoke({"input": user_input})
                raw_answer = response.get("answer", "No response")
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)