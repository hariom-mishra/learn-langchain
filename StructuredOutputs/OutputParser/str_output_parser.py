from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

template1 = PromptTemplate(
    template="Write a detailed report on {topic}?",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({"topic": "black holes"})
print("Final Result:\n", res)