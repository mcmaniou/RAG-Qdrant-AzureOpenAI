from flask import Flask, request, jsonify, render_template
from search_db import answer_question

app = Flask(__name__)


# API endpoint
@app.route('/api/answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    question = data['question']
    answer = answer_question(question)
    return jsonify({'answer': answer})


# UI endpoint
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
