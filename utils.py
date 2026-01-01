from tensorflow.keras import callbacks

class TrainLogger(callbacks.Callback):
    def __init__(self, interval=5):
        super(TrainLogger, self).__init__()
        self.interval = interval
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))
        if epoch % self.interval == 0:
            print(f'epoch={epoch} loss={logs["loss"]:.4f} acc={logs["accuracy"]:.4f}')


class TestLogger(callbacks.Callback):
    def __init__(self, X_val,y_val,interval=5):
        super(TestLogger, self).__init__()
        self.interval = interval
        self.X_val = X_val
        self.y_val = y_val
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        self.losses.append(loss)
        self.accs.append(acc)
        if epoch % self.interval == 0:
            print(f'[Test] epoch={epoch} loss={loss:.4f} acc={acc:.4f}')
