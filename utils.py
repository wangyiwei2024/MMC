class EarlyStopping:
    def __init__(self, patience=6):
        self.best = 0.0
        self.wait = 0
        self.patience = patience

    def step(self, score):
        if score > self.best:
            self.best = score
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience
