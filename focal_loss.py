import torch

def focal_loss(y_pred, y_true, alpha = 0.25, gamma = 2, reduction = 'mean'):
	'''
	focal_loss = -alpha * (1 - pt) ^ gamma * log(pt)

	y_pred/y_true: [batch, H, W]
	'''

	y_pred = y_pred.reshape(y_pred.shape[0], -1)
	y_true = y_true.reshape(y_true.shape[0], -1)

	pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

	if reduction == 'sum':
		focal_loss = torch.sum(-alpha * (1 - pt) ** gamma * torch.log(pt))
	elif reduction == 'mean':
		focal_loss = torch.mean(-alpha * (1 - pt) ** gamma * torch.log(pt))
	else:
		raise Error('Wrong reduction defined for focal loss')

	return focal_loss

if __name__ == "__main__":

	y_pred = torch.rand((3,400,538))
	y_true = torch.rand((3,400,538))

	fl = focal_loss(y_pred, y_true)
	print(fl)