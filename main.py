import torch
from torch.autograd import Variable
from torch.nn import functional as F
from preproc import *
from blur import genBlur, genSeq
from TVloss import TVloss
import time

# settings
device = torch.device('cuda:0')

# get image
im_size = (256, 256)  # for real blurred images, im_size should be set to images' size.
blur = get_image('./images/syn-affine-blur.png', im_size, show=False)

"""
# add noise
snr = 30
std = blur.std().item()
noise = np.random.normal(loc=0, scale=np.sqrt(std ** 2 / (10 ** (snr / 10))), size=blur.size())
blur = blur + torch.from_numpy(noise.astype('float32'))
utils.save_image(blur[0], './results/blur-noise.png')
"""

# init params
N = 7
itN = 5000
latent_ = blur.to(device)
# latent_ = torch.zeros(blur.size()).to(device)
latent_ = Variable(latent_, requires_grad=True)
blur = blur.to(device)
blur = Variable(blur, requires_grad=True)

A_ = torch.eye(2).to(device)
T_ = (torch.zeros(2, 1)+0.1*torch.rand(2, 1)).to(device)
A_, T_ = Variable(A_, requires_grad=True), Variable(T_, requires_grad=True)
I = torch.eye(2).to(device)
O = torch.zeros(2, 1).to(device)
optimizer1 = torch.optim.Adam([A_, T_], lr=0.01)
optimizer2 = torch.optim.Adam([latent_], lr=0.03)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1000, gamma=1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1000, gamma=0.95)
TVloss = TVloss().to(device)
TVweight = 0.025
"""
syn-blur: TVweight = 0.025
real-blur: TVweight = 0.04 ~ 0.1
"""
Aweight = 10
Tweight = 1
temp = 0

start = time.time()
for it in range(itN):
    # A_, T_ sub-problem
    if A_.grad is not None:
        A_.grad.zero_()
    if T_.grad is not None:
        T_.grad.zero_()
    blur_1 = genBlur(A_, T_, latent_, N)
    # loss backward
    loss1 = F.l1_loss(blur_1, blur) + Aweight*F.mse_loss(A_, I) + Tweight*F.mse_loss(T_, O)
    loss1.backward()
    optimizer1.step()
    scheduler1.step()

    # latent_ sub-problem
    if latent_.grad is not None:
        latent_.grad.zero_()
    blur_2 = genBlur(A_, T_, latent_, N)
    # loss backward
    l1loss = F.l1_loss(blur_2, blur)
    tvloss = TVloss(latent_)
    loss2 = l1loss + TVweight*tvloss
    loss2.backward()
    optimizer2.step()
    scheduler2.step()

    if it % 500 == 0:
        print('iteration: %d, l1loss: %.5f, tv-l1: %.5f' % (it, l1loss, tvloss))

    # if (it >= 3000) & (abs(loss2.detach().cpu().numpy() - temp) <= 1e-5):
    #     break
    # temp = loss2.detach().cpu().numpy()

end = time.time()
print("running time:%.2f s" % (end-start))

# save results
print('A=', A_)
print('T=', T_)
genSeq(A_, T_, latent_, N)

