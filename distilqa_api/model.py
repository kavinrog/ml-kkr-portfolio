# model.py
from transformers import pipeline

# Load model once
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def answer_question(question: str, context: str) -> str:
    result = qa_pipeline(question=question, context=context)
    return result['answer']