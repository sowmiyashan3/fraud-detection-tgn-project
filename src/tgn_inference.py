import torch

def run_tgn_inference(memory, classifier, test_df, prepare_tgn_data, node2id):
    test_src, test_dst, test_t, test_m, test_y = prepare_tgn_data(test_df, node2id)
    memory.reset_state()
    memory.eval()
    classifier.eval()

    test_predictions = []

    print("Running TGN inference on test set...")
    with torch.no_grad():
        for i in range(len(test_src)):
            if i % 5000 == 0:
                print(f"  Processed {i}/{len(test_src)} transactions...")

            z_s, _ = memory(test_src[i:i+1])
            z_d, _ = memory(test_dst[i:i+1])

            logit = classifier(z_s, z_d)
            prob = torch.sigmoid(logit).item()
            test_predictions.append(prob)

            memory.update_state(
                test_src[i:i+1],
                test_dst[i:i+1],
                test_t[i:i+1],
                test_m[i:i+1]
            )

    test_df_labeled = test_df[test_df['label'] != -1].copy()
    test_df_labeled['fraud_score'] = test_predictions

    return test_df_labeled
