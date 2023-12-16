import torch


class Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k_val):
        return torch.where(scores > k_val, torch.tensor([0.]).to(scores.device), torch.tensor([1.]).to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None


class SoftBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k_val):
        return torch.where(scores > k_val, torch.tensor([0.]).to(scores.device), scores)

    @staticmethod
    def backward(ctx, g):
        return g, None,

class IdentityMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k_val):
        return scores

    @staticmethod
    def backward(ctx, g):
        return g, None,


def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()
