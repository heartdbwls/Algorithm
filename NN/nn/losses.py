from NN.autograd import Tenosr
from NN.nn.activations import Softmax
import NN as nnet
import numpy as np

class MSELoss(Tensor):
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        return (y_pred.sub(y_true)).power(2).sum().div(np.prod(y_pred.data.shape))

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class CrossEntropyLoss(Tensor):
    def __init__(self, weight = None, ignore_index = -100, reduction = "mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.softmax = Softmax(axis=1)
        self.nll_loss = NLLLoss(weight, ignore_index, reduction)

    def forward(self, y_pred, y_true):
        y_pred = self.softmax(y_pred).log()
        return self.nll_loss(y_pred, y_true)

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
