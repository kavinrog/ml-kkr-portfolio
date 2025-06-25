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

def get_model_type():
    """
    Returns the type of the model.
    This function is useful for testing purposes.
    """
    return "transformers"
def get_model_version():
    """
    Returns the version of the model.
    This function is useful for testing purposes.
    """
    return "v1.0.0"
def get_model_description():
    """
    Returns a description of the model.
    This function is useful for testing purposes.
    """
    return "DistilBERT model fine-tuned on the SQuAD dataset for question answering tasks."
def get_model_metadata():
    """
    Returns metadata about the model.
    This function is useful for testing purposes.
    """
    return {
        "name": get_model_name(),
        "type": get_model_type(),
        "version": get_model_version(),
        "description": get_model_description()
    }

def get_model_info():
    """
    Returns a dictionary with model information.
    This function is useful for testing purposes.
    """
    return {
        "name": get_model_name(),
        "type": get_model_type(),
        "version": get_model_version(),
        "description": get_model_description(),
        "metadata": get_model_metadata()
    }
    
def is_model_loaded():
    """
    Checks if the model pipeline is loaded.
    Returns True if loaded, False otherwise.
    """
    return qa_pipeline is not None