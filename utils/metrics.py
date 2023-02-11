import torch


def si_log(prediction, target):
    # Make sure predictions are higher than 0
    prediction = torch.clamp(prediction, min=1e-6)
    mask = ~torch.isnan(target)
    d = torch.log(prediction[mask]) - torch.log(target[mask])
    n = torch.numel(mask)
    si_log = torch.sum(torch.square(d), dim=0) / n - torch.sum(d, dim=0) ** 2 / n ** 2
    return torch.sqrt(si_log) * 100.0


def mae(prediction, target):
    mask = ~torch.isnan(target)
    return torch.mean((prediction[mask] - target[mask]).abs())


def rsme(prediction, target):
    mask = ~torch.isnan(target)
    rmse = (target[mask] - prediction[mask]) ** 2
    return torch.sqrt(rmse.mean())


def sq_rel(prediction, target):
    mask = ~torch.isnan(target)
    sq_rel = torch.mean(((target[mask] - prediction[mask]) ** 2) / target[mask])
    return sq_rel


def abs_rel(prediction, target):
    mask = ~torch.isnan(target)
    return torch.mean(torch.abs(target[mask] - prediction[mask]) / target[mask])
