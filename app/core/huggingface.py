from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def rerank_docs(user_question, retriever, top=10):
    # Rerank top 10 documents in the 15 retrieved documents
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cpu"})
    compressor = CrossEncoderReranker(model=model, top_n=top)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    reranked_docs = compression_retriever.invoke(user_question)
    return reranked_docs


def huggingface_embedding_model(model_name="vinai/phobert-base"):
    model_name = model_name
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf
