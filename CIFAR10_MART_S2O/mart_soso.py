import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(size_average=False)
    
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    output = model(x_natural)
    output_adv = model(x_adv)

    adv_probs = F.softmax(output_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(output_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(output, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    
    device = torch.device("cuda")
    with torch.no_grad():
        aa = torch.zeros(output.size()[1], output.size()[1]).to(device)
        for xx in output:
            aa += torch.kron(xx, xx.reshape((-1,1)))
        #aa = torch.abs(torch.inverse(aa))
        aa = -1*torch.abs(aa)
        aa_min, aa_indexes = torch.min(aa, 1)
        aa = aa-aa_min.reshape(-1,1)
        for i in range(len(aa)):
            aa[i,i]=0.
        aa = 0.8*aa/aa.sum(1).reshape(-1,1)
        for i in range(len(aa)):
            aa[i,i]=0.2
        yy = aa[y]

        aa_adv = torch.zeros(output_adv.size()[1], output_adv.size()[1]).to(device)
        for xx in output_adv:
            aa_adv += torch.kron(xx, xx.reshape((-1,1)))
        #aa_adv = torch.abs(torch.inverse(aa_adv))
        aa_adv = -1*torch.abs(aa_adv)
        aa_adv_min, aa_adv_indexes = torch.min(aa_adv, 1)
        aa_adv = aa_adv-aa_adv_min.reshape(-1,1)
        for i in range(len(aa_adv)):
            aa_adv[i,i]=0.
        aa_adv = 0.8*aa_adv/aa_adv.sum(1).reshape(-1,1)
        for i in range(len(aa_adv)):
            aa_adv[i,i]=0.2
        yy_adv = aa_adv[y]

    loss_yy = (1.0 / yy.size()[0]) * criterion_kl(F.log_softmax(model(x_natural), dim=1),yy)
    loss_yy_adv = (1.0 / yy_adv.size()[0]) * criterion_kl(F.log_softmax(model(x_adv), dim=1),yy_adv)
    
    loss = loss_adv*0.7 + 0.3*0.5*(loss_yy+loss_yy_adv) + float(beta) * loss_robust

    return loss
