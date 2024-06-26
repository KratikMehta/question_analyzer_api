import json
import os
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ["OPENAI_API_KEY"])


class Question(BaseModel):
    question: str
    category: List[str]


@app.post("/question")
async def post_question(question: Question) -> dict[str, Any]:
    prompt = (
        f"You are an intelligent assistant. Your task is to assess a customer's question and provide the following:\n\n"
        "Urgency Level: Determine the urgency level of the question and categorize it into one of three levels: low, medium, or high urgency.\n\n"
        "High Urgency: The question involves a critical issue that needs immediate attention.\n"
        "Medium Urgency: The question pertains to important matters that need prompt but not immediate attention.\n"
        "Low Urgency: The question concerns routine inquiries, general information, or minor issues that do not require immediate action.\n\n"
        f"Category: Classify the question into one of the following categories: {', '.join(question.category)}\n"
        "Keyword/Phrase: Identify a keyword or phrase that best describes the main topic of the question.\n\n"
        f"Return result in below JSON format:\n"
        'category: "",\n'
        'urgency: "",\n'
        'keyword: ""\n\n'
        f"Question is: {question.question}"
    )

    try:
        response = llm.invoke(prompt).content
        json_string = response.strip("```json").strip().strip("```")
        return json.loads(json_string)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error processing the question response"
        ) from e
