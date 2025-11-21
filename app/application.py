from flask import Flask, render_template, request, redirect, url_for, session
from app.components.retriever import get_retriever_qa
from dotenv import load_dotenv
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
    # Initialize session messages
    if "messages" not in session:
        session["messages"] = []
    error_msg = None  
    if request.method == 'POST':
        question = request.form.get('prompt')
        if question:
            messages = session["messages"]
            messages.append({"role": "user", "content": question})
            session["messages"] = messages

            try:
                qa_chain = get_retriever_qa()
                
                # Convert session messages to chat_history
                # Convert session messages to chat_history tuples
                chat_history = [
                    (msg['content'], messages[i+1]['content'])
                    for i, msg in enumerate(messages)
                    if msg['role'] == 'user' and i+1 < len(messages) and messages[i+1]['role'] == 'assistant'
                ]

                temp_user = None
                for msg in messages:
                    if msg["role"] == "user":
                        temp_user = msg["content"]
                    elif msg["role"] == "assistant" and temp_user is not None:
                        chat_history.append((temp_user, msg["content"]))
                        temp_user = None

                user_question = question
                result = qa_chain({"question": user_question, "chat_history": chat_history})
                answer = result['answer']  # or 'result', depending on chain

                
                messages.append({"role": "user", "content": user_question})
                messages.append({"role": "assistant", "content": answer})
                session["messages"] = messages


            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                messages.append({"role": "assistant", "content": error_msg})
                session["messages"] = messages
    return render_template('index.html', messages=session["messages"], error=error_msg)
    


@app.route('/clear', methods=['GET'])
def clear_history():
    session.pop("messages", None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
