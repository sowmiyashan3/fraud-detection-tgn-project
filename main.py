import torch
import torch.optim as optim

from src.config import *
from src.data_loader import load_transaction_data, load_identity_data, merge_data
from src.feature_engineering import engineer_features
from src.graph_builder import build_graph
from src.model import memory, classifier, prepare_tgn_data
from src.train import train_epoch, eval_epoch
from src.tgn_inference import run_tgn_inference
from src.neo4j_graph import FraudGraphStore
from src.pinecone_index import setup_pinecone_index
from src.rag_llamaindex import FraudRAGLlamaIndex
from src.rag_langchain import FraudRAGLangChain

print("Loading and preprocessing data...")

transaction_df = load_transaction_data()
identity_df = load_identity_data()

df = merge_data(transaction_df, identity_df)
df, scaler = engineer_features(df)

edges_df = build_graph(df)
edges_df = edges_df.sort_values("timestamp")

print("Building node mappings...")

all_nodes = set(edges_df['src']).union(edges_df['dst'])
node2id = {node: idx for idx, node in enumerate(sorted(all_nodes))}

print("Creating temporal train/validation split...")

split_time = edges_df["timestamp"].quantile(0.8)

train_df = edges_df[edges_df["timestamp"] <= split_time]
val_df   = edges_df[edges_df["timestamp"] > split_time]

train_src, train_dst, train_t, train_m, train_y = prepare_tgn_data(train_df, node2id)
val_src, val_dst, val_t, val_m, val_y = prepare_tgn_data(val_df, node2id)

print("Training TGN model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
memory = memory.to(device)
classifier = classifier.to(device)

optimizer = optim.Adam(
    classifier.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)


NUM_EPOCHS = 10
best_val_auc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_auc, train_ap = train_epoch(
        train_src, train_dst, train_t, train_m, train_y,
        memory, classifier, optimizer,
        batch_size=512,
        device=device,
    )

    val_auc, val_ap = eval_epoch(
        val_src, val_dst, val_t, val_m, val_y,
        memory, classifier,
        batch_size=512,
        device=device
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train AUC: {train_auc:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(memory.state_dict(), "best_memory.pt")
        torch.save(classifier.state_dict(), "best_classifier.pt")

print("Running TGN inference...")

memory.load_state_dict(torch.load("best_memory.pt"))
classifier.load_state_dict(torch.load("best_classifier.pt"))

memory.eval()
classifier.eval()

test_df_labeled = run_tgn_inference(
    memory,
    classifier,
    edges_df,
    prepare_tgn_data,
    node2id
)

print("Storing results in Neo4j...")

neo4j_store = FraudGraphStore(
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD
)

if neo4j_store.driver:
    neo4j_store.create_constraints()
    neo4j_store.load_transaction_graph(
        test_df_labeled,
        test_df_labeled["fraud_score"].tolist()
    )

print("Initializing Graph-RAG pipeline...")

index, embedding_model = setup_pinecone_index()

llama_rag = FraudRAGLlamaIndex(index)
langchain_rag = FraudRAGLangChain(index)

print("Generating explanations for high-risk transactions...\n")

high_risk_transactions = test_df_labeled.nlargest(3, "fraud_score")

for _, row in high_risk_transactions.iterrows():
    report = langchain_rag.comprehensive_report(
        row["src"],
        neo4j_store,
        row["fraud_score"]
    )
    print(report)
    print("=" * 80)
