class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.min_loss = None
        self.delta = delta

    def __call__(self, val_loss):
        if self.min_loss is None:
            self.min_loss = val_loss
            return False
        elif val_loss < (self.min_loss - self.delta):
            self.min_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
