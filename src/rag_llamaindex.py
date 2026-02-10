from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import GOOGLE_API_KEY

class FraudRAGLlamaIndex:
    """LlamaIndex + Gemini RAG for fraud analysis"""

    def __init__(self, pinecone_index):
        if not pinecone_index:
            self.query_engine = None
            return

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        Gemini.api_key = GOOGLE_API_KEY
        self.query_engine = self.index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")

    def explain_fraud(self, query):
        if not self.query_engine:
            return "RAG system not initialized"
        return str(self.query_engine.query(query))
