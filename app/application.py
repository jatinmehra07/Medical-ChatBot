from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from markupsafe import Markup
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# Initialize QA chain once at startup
qa_chain = create_qa_chain()

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                if qa_chain is None:
                    # Retry loading in case it was initialized before vectorstore existed
                    qa_chain = create_qa_chain()
                
                if qa_chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")

                # Modern LangChain input/output key mapping
                response = qa_chain.invoke({"input": user_input})
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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)