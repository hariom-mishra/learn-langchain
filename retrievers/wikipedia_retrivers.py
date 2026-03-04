from langchain_community.retrievers import WikipediaRetriever;

ret = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

docs = ret.invoke(query)
print(docs)