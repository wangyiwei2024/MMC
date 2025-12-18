import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MMCDataset(Dataset):
    def __init__(self, root, label_file=None, train=True):
        self.root = root
        self.train = train
        self.samples = []

        if label_file is not None:
            with open(label_file, 'r') as f:
                for line in f:
                    name, label = line.strip().split()
                    self.samples.append((name, int(label)))
        else:
            files = sorted(os.listdir(os.path.join(root, 'color')))
            self.samples = [(f, -1) for f in files]

        self.tf_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

        self.tf_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        tf = self.tf_train if self.train else self.tf_test

        def load(mod):
            path = os.path.join(self.root, mod, name)
            return tf(Image.open(path).convert('RGB'))

        return load('color'), load('depth'), load('infrared'), label, name
