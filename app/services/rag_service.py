from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pydantic import SecretStr

from app.core import llm_model
from app.core.config import settings
from app.core.huggingface import huggingface_embedding_model, rerank_docs

api_key = settings.PINECONE_API_KEY
index_name = settings.PINECONE_INDEX_NAME
# namespace = settings.PINECONE_NAMESPACE
google_api_key = settings.GOOGLE_API_KEY


class RAGServices:
    def __init__(self, data):
        self.data = data

    def get_rag(self, user_question):

        # embeddings = huggingface_embedding_model()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=SecretStr(google_api_key))

        new_db = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)

        retriever = new_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        # docs1 = new_db.similarity_search(query=user_question, k=15) # Đây là cách cũ
        # docs1 = retriever.invoke(user_question) Đây là cách mới

        # Rerank top 10 documents in the 15 retrieved documents
        reranked_docs = rerank_docs(user_question, retriever)

        docs2 = llm_model.filter_by_metadata(user_question, new_db)

        docs = reranked_docs + docs2

        metadata_combined = "\n".join([str(doc.metadata) for doc in docs])

        chain = llm_model.get_conversational_chain()

        response = chain({"input_documents": docs,
                          "metadata": metadata_combined,
                          "question": user_question},
                         return_only_outputs=True)

        return response["output_text"]
