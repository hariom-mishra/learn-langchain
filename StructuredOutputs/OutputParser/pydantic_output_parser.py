from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation", # Required for the base class
    provider="auto",        # CRITICAL: Let HF choose the active provider
    max_new_tokens=512,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=0, lt=120, description="The age of the person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Give me the name, age and city of a fictional person\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.format()


model_response = model.invoke(prompt)

print(model_response.content)