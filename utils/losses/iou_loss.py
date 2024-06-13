import torch
from torch import nn

from utils.evaluation.multi_channel_iou import IoU


class IoULoss(nn.Module):
    """ Intersection over union loss.
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Intersection over union loss.
        """
        super().__init__(*args, **kwargs)

        self.iou = IoU()

    def forward(self, x: torch.TensorType, y: torch.TensorType):
        """ Calculates the IoU of two masks with multiple channels and returns a vector with
        1 - mean IoU of channels.
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Reference tensor.
        Returns:
            torch.Tensor: Vector with IoUs.
        """     
        iou_loss = self.iou(x, y)
        iou_loss = torch.mean(iou_loss, dim=-1)
        iou_loss = iou_loss

        return iou_loss

if __name__ == "__main__":
    x = torch.ones((3, 3, 3))
    y = torch.ones((3, 3, 3))
    iou = IoU()
    print(iou(x, y))
