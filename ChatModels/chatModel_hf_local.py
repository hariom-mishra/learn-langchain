import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# 1. Initialize the local model correctly
# We use from_model_id so LangChain "remembers" which model this is
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    ),
    # Crucial for your M4 Mac:
    device_map="auto", 
    model_kwargs={"torch_dtype": torch.bfloat16} 
)

# 2. Initialize ChatHuggingFace
# It will now automatically pull the Gemma tokenizer to format the prompt
model = ChatHuggingFace(llm=llm)

# 3. Use it!
response = model.invoke("2 + 2 = ?")
print(response.content)