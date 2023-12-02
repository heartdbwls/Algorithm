import cupy as cp

def sigmoid(z, a=None, derivative=False):
	if derivative:
		return a * (1 - a)
	else:
		return 1.0 / (1 + cp.exp(-z.clip(-88.72283, 88.72283)))

def relu(z, a=None, derivative=False):
	if derivative:
		return z > 0
	else:
		# z[z<0]=0
		# return z
		# return z*(z>0)
		return cp.maximum(0, z)


def softmax(z, a=None, derivative=False):
	if derivative:
		# a1*(1-a1)-a1a2
		return 1
	else:
		exps = cp.exp(z - cp.max(z, axis=1, keepdims=True))
		# return exps/cp.sum(exps, axis=1, keepdims = True)
		exps /= cp.sum(exps, axis=1, keepdims=True)
		return exps


def cross_entropy(outputs, labels, epsilon=1e-12):
	labels = labels.clip(epsilon, 1 - epsilon)
	outputs = outputs.clip(epsilon, 1 - epsilon)
	return -labels * cp.log(outputs) - (1 - labels) * cp.log(1 - outputs)


def del_cross_soft(outputs, labels):
	return (outputs - labels)


def mean_squared_error(outputs, labels):
	return ((outputs - labels) ** 2) / 2


def del_mean_squared_error(outputs, labels):
	return (outputs - labels)


def echo(z, a=None, derivative=False, **kwargs):
	return z
