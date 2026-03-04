from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, TypedDict, Annotated
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    sentiment: str = Field(description="return sentiment of the review either positive, negative or neutral")
    summary: str = Field(description="A brief summary of the review")
    cons: Optional[list[str]] = Field(description="List of negative aspects mentioned in the review")

structured_model = model.with_structured_output(Review)

res = structured_model.invoke("the product is great overhaul, I loved the new features and improvements.")
print(res.sentiment)
print(res.summary)
print(res.cons)
