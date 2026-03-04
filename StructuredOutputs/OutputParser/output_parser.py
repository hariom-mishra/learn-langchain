from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


model = ChatOpenAI()

template1 = PromptTemplate(
    template="Write a detailed report on {topic}?",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "black holes"})

result = model.invoke(prompt1)
print("Generated Report:\n", result.content)

prompt2 = template2.invoke({"text": result.content})

res = model.invoke(prompt2)
print("Generated Summary:\n", res.content)