from NN.autograd import Tensor
import numpy as np


class _SigmoidTensor(Tensor): #Static sigmoid tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
       
        self.args[0].backward(grad * self.data * (1 - self.data))

class Sigmoid(): #Static sigmoid computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = 1 / (1 + np.exp(-x.data))
        return _SigmoidTensor(f_x, [x], "sigmoid")

    def __call__(self, x):
        return self.forward(x)

class _ReLUTensor(Tensor): #Static ReLU tensor for backpropagation
    def __init__(self, data, args, op):
        super().__init__(data, args, op)

    def backward(self, grad=1):
        self.args[0].backward(grad * (self.data > 0))

class ReLU(): #Static ReLU computation
    def __init__(self):
        pass

    def forward(self, x):
        f_x = np.maximum(0, x.data)
        return _ReLUTensor(f_x, [x], "relu")

    def __call__(self, x):
        return self.forward(x)


class Softmax(): #Dynamic Softmax computation
    def __init__(self, axis = 1):
        self.axis = axis

    def forward(self, x):
        e_x = x.sub(x.max(axis=self.axis, keepdims=True)).exp()
        return e_x.div(e_x.sum(axis=self.axis, keepdims=True))

    def __call__(self, x):
        return self.forward(x)


activations= {
    "sigmoid": Sigmoid(),
    "softmax": Softmax(),
    "relu": ReLU(),  
}
