from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

'''
this file is used to generate embeddings for the given document
'''

load_dotenv()

document = [
    "What is the capital of India?",
    "What is the largest mammal in the world?",
    "What is the boiling point of water?"
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result = embeddings.embed_documents(document)

print(result)