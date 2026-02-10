from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

def setup_pinecone_index(index_name="fraud-detection"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name in [idx['name'] for idx in pc.list_indexes()]:
        pc.delete_index(index_name)
        time.sleep(10)
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    index = pc.Index(index_name)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, embedding_model
