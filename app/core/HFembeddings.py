from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


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


def reranker(retriever, user_question):
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": "cpu"})
    compressor = CrossEncoderReranker(model=model, top_n=15)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    compressed_docs = compression_retriever.invoke(user_question)
    return compressed_docs
