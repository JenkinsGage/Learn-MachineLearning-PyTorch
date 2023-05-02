import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FourierLoss(nn.Module):
    def __init__(self, fft_topk=8, p=2):
        super(FourierLoss, self).__init__()
        self.fft_topk = fft_topk
        self.p = p

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        output_rfft = torch.abs(torch.fft.rfft(output, dim=1, norm='ortho'))
        target_rfft = torch.abs(torch.fft.rfft(target, dim=1, norm='ortho'))

        _, topk_index = torch.topk(target_rfft, self.fft_topk, dim=1)
        mask = torch.zeros_like(target_rfft)
        mask.scatter_(dim=1, index=topk_index, value=1.0)
        distance_rfft = torch.abs(output_rfft - target_rfft) * mask + output_rfft * (1 - mask)
        distance_rfft = torch.sum(distance_rfft**self.p, dim=1)**(1/self.p)
        return torch.mean(distance_rfft)
