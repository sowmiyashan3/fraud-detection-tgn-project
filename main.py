import torch
from src.config import FEATURE_COLS, OUT_PATH
from src.utils import cleanup_memory
from src.data_loader import load_transaction_data, load_identity_data, merge_data
from src.feature_engineering import engineer_features
from src.graph_builder import build_graph
from src.dataset import prepare_data
from src.model import AttentionClassifier, FocalLoss, build_memory
from src.train import train_epoch, eval_epoch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
print("="*70)
print(f"Device: {device}")
if use_gpu:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*70)

transaction_df = load_transaction_data(sample_frac=0.4)
identity_df = load_identity_data()
df = merge_data(transaction_df, identity_df)
df, scaler = engineer_features(df)
edges_df = build_graph(df)

n_total = len(edges_df)
n_train = int(0.56 * n_total)
n_val = int(0.14 * n_total)

train_df = edges_df.iloc[:n_train].copy()
val_df = edges_df.iloc[n_train:n_train+n_val].copy()
test_df = edges_df.iloc[n_train+n_val:].copy()

all_nodes = set()
for df_split in [train_df, val_df, test_df]:
    all_nodes.update(df_split['src'].unique())
    all_nodes.update(df_split['dst'].unique())

node2id = {node: i for i, node in enumerate(sorted(all_nodes))}
num_nodes = len(node2id)
print(f"Nodes: {num_nodes}")

train_src, train_dst, train_t, train_m, train_y = prepare_data(train_df, node2id, neg_ratio=0.2)
val_src, val_dst, val_t, val_m, val_y = prepare_data(val_df, node2id, neg_ratio=0.2)

memory = build_memory(num_nodes, len(FEATURE_COLS))
classifier = AttentionClassifier(memory_dim=100)
if use_gpu:
    classifier = classifier.to(device)
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

optimizer = AdamW(list(memory.parameters()) + list(classifier.parameters()), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

EPOCHS = 30
best_val_auc = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_auc, train_ap = train_epoch(train_src, train_dst, train_t, train_m, train_y,
                                                  memory, classifier, optimizer, batch_size=512,
                                                  device=device, focal_loss=focal_loss)
    val_auc, val_ap = eval_epoch(val_src, val_dst, val_t, val_m, val_y, memory, classifier, batch_size=512, device=device)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    gap = train_auc - val_auc

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save({
            'memory': memory.state_dict(),
            'classifier': classifier.state_dict(),
            'val_auc': val_auc
        }, f"{OUT_PATH}/best_model_v4.pt")
        print(f"✅ E{epoch:02d} | Loss: {train_loss:.4f} | Train: {train_auc:.4f} | Val: {val_auc:.4f} | Gap: {gap:.4f} ⭐")
    else:
        patience_counter += 1
        print(f"E{epoch:02d} | Loss: {train_loss:.4f} | Train: {train_auc:.4f} | Val: {val_auc:.4f} | Gap: {gap:.4f}")

    if epoch % 5 == 0:
        cleanup_memory()

    if patience_counter >= 8:
        print(f"⚠️  Early stop")
        break

print(f"🎯 BEST VALIDATION AUC: {best_val_auc:.4f}")
cleanup_memory()
