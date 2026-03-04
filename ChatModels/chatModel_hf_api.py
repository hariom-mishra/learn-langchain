import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

# 1. API Token Setup
# Make sure your token has "Inference Providers" permissions enabled in HF settings
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# 2. Define the Model with "auto" provider
# DeepSeek-R1 models are currently the most reliable for the free tier
repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation", # Required for the base class
    provider="auto",        # CRITICAL: Let HF choose the active provider
    max_new_tokens=512,
    repetition_penalty=1.03,
)

# 3. Wrap in ChatHuggingFace
# This handles the internal conversion to the 'conversational' task
chat_model = ChatHuggingFace(llm=llm)

# 4. Invoke the model
messages = [
    HumanMessage(content="what is the capital of india?")
]

try:
    response = chat_model.invoke(messages)
    print("AI Response:", response.content)
except Exception as e:
    print(f"Error: {e}")