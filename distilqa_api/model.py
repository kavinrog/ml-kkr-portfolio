# model.py
from transformers import pipeline

# Load model once
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def answer_question(question: str, context: str) -> str:
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def get_model():
    """
    Returns the loaded model.
    This function is useful for testing purposes.
    """
    return qa_pipeline      

def get_model_name():
    """
    Returns the name of the model.
    This function is useful for testing purposes.
    """
    return "distilbert-base-uncased-distilled-squad"

