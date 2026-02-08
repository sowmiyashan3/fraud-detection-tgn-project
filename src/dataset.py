import torch
import numpy as np

def prepare_data(df, node2id, neg_ratio=0.2):
    df_pos = df[df['label'] != -1].copy()

    src = torch.tensor([node2id[x] for x in df_pos['src']], dtype=torch.long)
    dst = torch.tensor([node2id[x] for x in df_pos['dst']], dtype=torch.long)
    t = torch.tensor(df_pos['timestamp'].values // 10**9, dtype=torch.long)

    msg_list = [feat if isinstance(feat, list) else [0.0]*len(df_pos['edge_feat'][0])
                for feat in df_pos['edge_feat']]
    m = torch.tensor(msg_list, dtype=torch.float)
    y = torch.tensor(df_pos['label'].values, dtype=torch.float)

    # Hard negatives
    n_neg = int(len(src) * neg_ratio)
    if n_neg > 0:
        time_bins = t // 3600
        neg_indices = np.random.choice(len(src), n_neg, replace=True)

        neg_src = src[neg_indices]
        neg_t = t[neg_indices]
        neg_m = m[neg_indices]

        neg_dst = []
        for idx in neg_indices:
            time_bin = time_bins[idx]
            same_time = (time_bins == time_bin).nonzero(as_tuple=True)[0]
            if len(same_time) > 1:
                random_idx = same_time[np.random.randint(len(same_time))]
                neg_dst.append(dst[random_idx])
            else:
                neg_dst.append(torch.randint(0, len(node2id), (1,))[0])

        neg_dst = torch.stack(neg_dst)
        neg_y = torch.zeros(n_neg, dtype=torch.float)

        src = torch.cat([src, neg_src])
        dst = torch.cat([dst, neg_dst])
        t = torch.cat([t, neg_t])
        m = torch.cat([m, neg_m])
        y = torch.cat([y, neg_y])

    return src, dst, t, m, y
