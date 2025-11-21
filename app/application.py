from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from app.components.retriever import get_retriever_qa
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import os

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace('\n', '<br>\n'))
app.jinja_env.filters['nl2br'] = nl2br

@app.route('/', methods=['GET', 'POST'])
def index():
    if "message" in session:
        session["message"] = []

    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            message = session["message"]
            message.append({"role": "user", "content": question})
            session["message"] = message
            try:
                qa_response = get_retriever_qa()
                response = qa_response.invoke(question, hf_token=HF_TOKEN, chat_history=message)
                result = response.get("result", "No response generated.")
                message.append({"role": "assistant", "content": result})
                session["message"] = message
            except Exception as e:  
                error_msg = f"An error occurred: {str(e)}"
                message.append({"role": "assistant", "content": error_msg})
                session["message"] = message
                return render_template('index.html', messages=session["message"], error=error_msg)

        return redirect(url_for('index'))       

    return render_template('index.html', messages=session["message"], error=None)

@app.route('/clear', methods=['POST'])
def clear_history():
    session.pop("message", None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=True)