import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator

class AttentionClassifier(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim * 2, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, z_src, z_dst):
        z = torch.cat([z_src, z_dst], dim=1)
        z_unsqueezed = z.unsqueeze(1)
        z_attn, _ = self.attn(z_unsqueezed, z_unsqueezed, z_unsqueezed)
        z_attn = z_attn.squeeze(1)
        z = z + z_attn
        return self.mlp(z).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def build_memory(num_nodes, feature_dim, memory_dim=100, time_dim=100):
    message_module = IdentityMessage(
        raw_msg_dim=feature_dim,
        memory_dim=memory_dim,
        time_dim=time_dim
    )
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=feature_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=message_module,
        aggregator_module=LastAggregator()
    )
    return memory
