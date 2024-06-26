import json
import os
from typing import Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

model = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)


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
        '{category: "",urgency: "",keyword: ""}\n\n'
        f"Question is: {question.question}"
    )

    try:
        message = HumanMessage(content=prompt)
        response = model.invoke([message]).content
        return json.loads(response)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error processing the question response"
        ) from e
