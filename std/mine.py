from torch import nn, Tensor
from torch.optim import SGD


# https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb

def mine_regularization(model: nn.Module, mine_network: nn.Module, x: Tensor, b: int):
	model_optimizer = SGD(params=model.parameters(), lr=1.0, momentum=0)
	mine_optimizer = SGD(params=mine_network.parameters(), lr=1.0, momentum=0, maximize=False)
	y_hat, ts, tc = model.forward_features(x)


