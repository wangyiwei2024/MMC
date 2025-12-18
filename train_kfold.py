import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from datasets import MMCDataset
from model import MMCModel
from evaluate import evaluate
from utils import EarlyStopping

device = 'cuda'
root = 'data/train_2k'
label_file = 'data/train_2k/train_labels.txt'

dataset = MMCDataset(root, label_file, train=True)
labels = [l for _, l in dataset.samples]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf.split(range(len(labels)), labels)):
    print(f'\n===== Fold {fold} =====')

    train_set = Subset(dataset, tr_idx)
    val_set   = Subset(dataset, va_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

    model = MMCModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    early = EarlyStopping(patience=6)

    for epoch in range(25):
        model.train()
        for x1, x2, x3, y, _ in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y  = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x1, x2, x3), y)
            loss.backward()
            optimizer.step()

        mAP = evaluate(model, val_loader, device)
        print(f'Fold {fold} | Epoch {epoch} | mAP = {mAP:.4f}')

        if early.step(mAP):
            print('Early stopping triggered')
            break

    torch.save(model.state_dict(), f'convnext_fold{fold}.pth')
