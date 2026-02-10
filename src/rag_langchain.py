from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GOOGLE_API_KEY

class FraudRAGLangChain:
    """LangChain + Gemini RAG system"""

    def __init__(self, pinecone_index):
        if not pinecone_index:
            self.qa_chain = None
            return

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = PineconeVectorStore(index=pinecone_index, embedding=self.embeddings, text_key="text")

        template = """You are an expert fraud detection analyst.

Context - Similar Fraud Cases:
{context}

Question: {question}

Analysis:"""
        prompt = PromptTemplate.from_template(template)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def investigate(self, query):
        if not self.qa_chain:
            return "RAG system not initialized"
        return self.qa_chain.invoke(query)
