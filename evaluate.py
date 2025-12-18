import torch
import numpy as np
from sklearn.metrics import average_precision_score

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for x1, x2, x3, y, _ in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)

        prob = torch.softmax(model(x1, x2, x3), dim=1)
        all_probs.append(prob.cpu().numpy())
        all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return average_precision_score(all_labels, all_probs, average='macro')
