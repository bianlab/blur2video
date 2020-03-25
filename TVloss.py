import torch


def getImsize(x):
    b, c = x.size()[0], x.size()[1]
    h, w = x.size()[2], x.size()[3]
    return b, c, h, w


def getTVmap(x, type):
    b, c, h, w = getImsize(x)
    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h - 1, :])
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w - 1])
    return h_tv, w_tv


class TVloss(torch.nn.Module):
    def __init__(self):
        super(TVloss, self).__init__()
    def forward(self, x):
        b, c, h, w = getImsize(x)
        count_h = (h - 1) * w
        count_w = h * (w - 1)
        h_tv, w_tv = getTVmap(x, self.type)
        h_tv, w_tv = h_tv.sum(), w_tv.sum()
        loss = 2*(h_tv/count_h + w_tv/count_w)/b
        return loss

