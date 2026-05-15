from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

'''
this file is used to generate embeddings for the given query
'''

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result = embeddings.embed_query("What is the capital of India?")

print(result)