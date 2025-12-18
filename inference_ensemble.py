import torch
import csv
from torch.utils.data import DataLoader

from datasets import MMCDataset
from model import MMCModel

device = 'cuda'

dataset = MMCDataset('data/test_1k', train=False)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

models = []
for i in range(5):
    m = MMCModel().to(device)
    m.load_state_dict(torch.load(f'convnext_fold{i}.pth'))
    m.eval()
    models.append(m)

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label_pred'])

    for x1, x2, x3, _, names in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)

        prob_sum = None
        for m in models:
            p = torch.softmax(m(x1, x2, x3), dim=1)
            prob_sum = p if prob_sum is None else prob_sum + p

        prob = prob_sum / len(models)
        preds = prob.argmax(dim=1).cpu().tolist()

        for n, p in zip(names, preds):
            writer.writerow([n, p])
