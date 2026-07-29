import torch
from torch import Tensor
import matplotlib.pyplot as plt


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6,plot = False):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    if input.dim() == 4:
        plot = False

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    if plot:
        fig, axes = plt.subplots(2,1,sharex=True,sharey=True)
        axes[0].imshow(input[0,0])
        axes[0].set_title("Input")

        axes[1].imshow(target[0,0])
        axes[1].set_title('target')
        fig.text(
            0.5, 0.05, "dice is {:.4f}".format(dice.mean() ),
            ha='center', fontsize=14,
            bbox=dict(facecolor='red', alpha=0.3)
        )
        plt.show()


    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
