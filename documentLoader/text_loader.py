from langchain_community.document_loaders import TextLoader 

loader = TextLoader("documentLoader/example.txt")

documents = loader.load()

print(documents)