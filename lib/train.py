from torch import optim
import torch


class TrainInGpu():
    def __init__(self, net):
        self.optimizer = optim.Adam(net.parameters(), lr=5e-4)

    def UpdateOnce(self, pred, target):
        self.optimizer.zero_grad()
        loss = torch.norm(pred - target, p=2, dim=1).sum().to("cuda")
        loss.backward()
        self.optimizer.step()
        return loss.item()
