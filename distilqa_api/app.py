# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from model import answer_question

app = FastAPI()

class QARequest(BaseModel):
    question: str
    context: str

@app.get("/")
def root():
    return {"message": "DistilQA API is live"}

@app.post("/predict")
def predict(request: QARequest):
    answer = answer_question(request.question, request.context)
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "healthy"}    

@app.get("/info")
def info():
    return {
        "name": "DistilQA API",
        "version": "1.0.0",
        "description": "A simple API for question answering using DistilBERT."
    }