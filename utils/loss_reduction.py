import torch


def median(loss_each, num_parts=4):
    sorted_losses, _ = torch.sort(loss_each)
    num_samples = len(sorted_losses)
    ignore_size = num_samples // num_parts
    loss_all = torch.mean(sorted_losses[ignore_size:-ignore_size])
    return loss_all
