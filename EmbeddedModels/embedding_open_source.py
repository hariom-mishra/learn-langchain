from langchain_huggingface import HuggingFaceEmbeddings

'''
this file is used to generate embeddings for the given query
using open source model
'''

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of India?"

res = embedding.embed_query(text)

print(res)
