import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def train_epoch(src, dst, t, m, y, memory, classifier, optimizer, batch_size=512, device='cpu', focal_loss=None):
    memory.reset_state()
    classifier.train()
    memory.train()
    total_loss = 0
    preds, labels = [], []
    n_samples = len(src)

    for i in range(0, n_samples, batch_size):
        batch_src = src[i:i+batch_size]
        batch_dst = dst[i:i+batch_size]
        batch_t = t[i:i+batch_size]
        batch_m = m[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        optimizer.zero_grad()
        z_s, _ = memory(batch_src)
        z_d, _ = memory(batch_dst)

        z_s = z_s.detach().to(device)
        z_d = z_d.detach().to(device)
        batch_y_gpu = batch_y.to(device)

        logits = classifier(z_s, z_d)
        loss = focal_loss(logits, batch_y_gpu)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()
        memory.update_state(batch_src, batch_dst, batch_t, batch_m)

        total_loss += loss.item() * len(batch_y)
        preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        labels.extend(batch_y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0
    ap = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0.0
    return total_loss / n_samples, auc, ap

@torch.no_grad()
def eval_epoch(src, dst, t, m, y, memory, classifier, batch_size=512, device='cpu'):
    memory.reset_state()
    classifier.eval()
    memory.eval()
    preds, labels = [], []
    n_samples = len(src)

    for i in range(0, n_samples, batch_size):
        batch_src = src[i:i+batch_size]
        batch_dst = dst[i:i+batch_size]
        batch_t = t[i:i+batch_size]
        batch_m = m[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        z_s, _ = memory(batch_src)
        z_d, _ = memory(batch_dst)
        z_s = z_s.to(device)
        z_d = z_d.to(device)

        logits = classifier(z_s, z_d)
        memory.update_state(batch_src, batch_dst, batch_t, batch_m)

        preds.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch_y.cpu().tolist())

    auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0
    ap = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0.0
    return auc, ap
