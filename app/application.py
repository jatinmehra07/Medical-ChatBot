import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for
from markupsafe import Markup

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        # Import only when the first question is sent, preventing boot-time memory spikes
        from app.components.retriever import create_qa_chain
        qa_chain = create_qa_chain()
    return qa_chain

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                chain = get_qa_chain()
                if chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")

                response = chain.invoke({"input": user_input})
                result = response.get("answer", "No response")

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error : {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)