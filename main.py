"""
Google Colab Example: https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb
"""
import json
import os
import re
import time

from model.result import Result
from model.comparer import Comparer
from model.document import Document, load_documents, rawtxt_to_document
from entity.question import Question, Option, AnsweringResponse
from model.question_answering import QuestionAnswering
from model.retriever import Retriever
from flask import stream_with_context, make_response
from flask_restful import Resource, Api
from flask import Flask, flash, request, redirect, url_for, jsonify

if __name__ == '__main__':
    retriever = Retriever()
    comparer = Comparer()
    qa = QuestionAnswering()
    retriever.load_documents(load_documents())
    retriever.combine_data()


def solve_question(question):
    query = re.sub(r"[._]{5,}", " what ", question.content)
    query = re.sub(r"\s+", " ", query.strip())
    print(f"Input question: {query}")
    best_qa_answer = None
    # best_qa_context = None
    contexts = retriever.search(query)

    print("> QA model:")
    for context in contexts:
        qa_answer = qa.answer(query, context.content)
        print(f"QA: '{qa_answer.content}', score: {round(qa_answer.score, 3)}")
        if best_qa_answer is None or best_qa_answer.score < qa_answer.score * 4 + context.score:
            best_qa_answer = qa_answer
            best_qa_answer.score = qa_answer.score * 4 + context.score
            # best_qa_context = context

    print("> Comparing options with best context:")
    best_comparer_answer = None
    for option in question.options:
        score = comparer.compare(option.content, contexts[0].content)
        print("Comparer: {}, score: {:.3f}".format(option.content, score))
        if best_comparer_answer is None or best_comparer_answer.score < score * 3 + contexts[0].score:
            best_comparer_answer = Result(score * 3 + contexts[0].score, option.key)

    best_answer = None
    if best_qa_answer.score >= best_comparer_answer.score:
        print("> QA method has higher score, comparing QA answer with options:")
        for option in question.options:
            score = comparer.compare(option.content, best_qa_answer.content)
            print("Comparer: {}, score: {:.3f}".format(option.content, score))
            if best_answer is None or best_answer.score < score:
                best_answer = Result(score, option.key)
    else:
        print("> Comparing method has higher score.")
        best_answer = best_comparer_answer

    question.answer = best_answer.content
    print(f" -> Answer: {question.answer}")
    return best_answer


# db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)


@app.route('/knowledge', methods=['POST'])
def upload_knowledge():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    document = Document(file.filename)
    if os.path.isfile(document.path_txt):
        return make_response(jsonify(message=f"knowledge {document.name} exists"), 400)
    print(f"creating...")
    document = rawtxt_to_document(file.stream, file.filename)
    print(f"created {document.path_txt} success, encoding...")
    success = retriever.encode(document)
    if not success:
        return make_response(jsonify(message=f"Encode {document.name} fail. Document deleted"), 500)
    print(f"encoded {document.path_pt} success")
    retriever.load_document(document)
    retriever.combine_data()
    return make_response(jsonify(message=f"Upload and encode {document.name} success"), 200)


@app.route('/knowledge', methods=['DELETE'])
def delete_knowledge():
    filename = request.form["name"]
    document = Document(filename)
    if not os.path.isfile(document.path_txt):
        return make_response(jsonify(message=f"knowledge {document.name} does not exists"), 400)
    retriever.remove(document)

    return make_response(jsonify(message=f"delete {document.name} success"), 200)


@app.route('/qa', methods=['POST'])
def qa_res():
    def question_respond():
        json_questions = json.loads(json.dumps(request.json))
        questions = []
        answers_response = []
        for json_question in json_questions:
            options = []
            for json_option in json_question['options']:
                option = Option(json_option['key'], json_option['content'])
                options.append(option)

            question = Question(
                json_question['qn'],
                json_question['content'],
                options)
            questions.append(question)

        for question in questions:
            solve_question(question)
            answer = AnsweringResponse(question.qn, question.answer)
            answers_response.append(answer)
            yield json.dumps(answer.__dict__)

        # json_string = json.dumps([ob.__dict__ for ob in answers_response])

        # return json_string

    return app.response_class(stream_with_context(question_respond()))


if __name__ == '__main__':
    app.run(port=5000)
