from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, TypedDict, Annotated

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    sentiment: Annotated[str, "return sentiment of the review either positive, negative or neutral"]
    summary: Annotated[str, "A brief summary of the review"]
    cons: Annotated[Optional[list[str]], "List of negative aspects mentioned in the review"]

structured_model = model.with_structured_output(Review)

res = structured_model.invoke("the product is great overhaul, I loved the new features and improvements.")
print(res['sentiment'])
print(res['summary'])
