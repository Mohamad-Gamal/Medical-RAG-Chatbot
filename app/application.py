from flask import Flask, render_template, request, redirect, url_for, session
from app.components.retriever import get_retriever_qa
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

from markupsafe import Markup

def nl2br(value):
    return Markup(value.replace('\n', '<br>\n'))

app.jinja_env.filters['nl2br'] = nl2br

qa_chain = None
try:
    qa_chain = get_retriever_qa()
    print("✅ QA chain initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize QA chain: {e}")
    traceback.print_exc()

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'messages' not in session:
        session['messages'] = []
    
    error = None
    
    if request.method == 'POST':
        user_input = request.form.get('prompt', '').strip()
        
        if user_input:
            session['messages'].append({'role': 'user', 'content': user_input})
            
            try:
                if qa_chain is None:
                    error = "Medical AI system is initializing. Please wait..."
                    assistant_response = error
                else:
                    # Use the correct invocation format for RetrievalQA
                    response = qa_chain.invoke({"query": user_input})
                    assistant_response = response.get('result', 'I could not generate a response.')
                
                session['messages'].append({'role': 'assistant', 'content': assistant_response})
                
            except Exception as e:
                error = "Sorry, I encountered an error. Please try again."
                session['messages'].append({'role': 'assistant', 'content': error})
                print(f"Error: {e}")
                traceback.print_exc()
            
            session.modified = True
            return redirect(url_for('index'))
    
    return render_template('index.html', 
                         messages=session['messages'], 
                         error=error)    


@app.route('/clear', methods=['GET'])
def clear_history():
    session.pop("messages", None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
