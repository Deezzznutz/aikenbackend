from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import torch
from typing import List

# Load model and tokenizer (for text formatting/embedding)
model = BertModel.from_pretrained("./my_trained_model")  # Assuming your fine-tuned model for text formatting
tokenizer = BertTokenizer.from_pretrained("./my_trained_model")

app = FastAPI()

# Define a class for a single question
class Question(BaseModel):
    question: str
    options: List[str]
    correct_answer: str  # Correct answer as a string

# Define a class for the request body (list of questions for the entire exam)
class ExamRequest(BaseModel):
    questions: List[Question]

@app.post("/predict/")
async def predict(request: ExamRequest):
    formatted_questions = []

    # Process each question in the exam
    for q in request.questions:
        question = q.question
        options = q.options
        correct_answer = q.correct_answer

        # Format the question and options text
        formatted_question = reformat_question(question, options, correct_answer)

        # Tokenize the question and options for text formatting/embedding
        encoded_question = tokenizer.encode_plus(
            question,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoded_options = [tokenizer.encode_plus(
            option,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ) for option in options]

        # Get the model's output for the question and each option (formatting/embedding)
        with torch.no_grad():
            question_embedding = model(**encoded_question).last_hidden_state.mean(dim=1)  # Average embedding of the question
            option_embeddings = [model(**encoded_option).last_hidden_state.mean(dim=1) for encoded_option in encoded_options]

        # Collect the formatted results for this question
        formatted_questions.append({
            "formatted_question": formatted_question,
            "question_embedding": question_embedding.squeeze().tolist(),
            "option_embeddings": [embedding.squeeze().tolist() for embedding in option_embeddings]
        })

    # Return all formatted questions with embeddings
    return {"questions": formatted_questions}

def reformat_question(question: str, options: list, correct_answer: str):
    """
    Reformat the question and options into the desired output format.
    """
    formatted_question = f"{question}\n"
    for i, option in enumerate(options, start=1):
        formatted_question += f"{chr(64+i)}. {option}\n"  # Option A, B, C, D

    # Format the answer correctly in the output
    formatted_answer = f"ANSWER: {chr(64 + options.index(correct_answer))}"  # Get the correct option letter
    formatted_question += formatted_answer

    return formatted_question
