import torch
from torch import nn

from utils.evaluation.pixelwise_iou import SingleChannelIoU
from utils.errors.exceptions import WrongTensorShapeException


class IoU(nn.Module):
    """ Intersection over union multi channel evaluation metric.
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Intersection over union multi channel evaluation metric.
        """
        super().__init__(*args, **kwargs)
        self.iou = SingleChannelIoU()

    def forward(self, x: torch.TensorType, y: torch.TensorType) -> torch.Tensor:
        """ Calculates the IoU of two masks with multiple channels and returns a vector with
        the mean IoU of channels in the first position and all IoUs for each channel.
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Reference tensor.
        Returns:
            torch.Tensor: Vector with IoUs.
        """
        if x.shape != y.shape:
            raise WrongTensorShapeException("Both tensors must have same shape")

        if x.dim() < 3:
            return self.iou(x, y)
        elif x.dim() == 3 and x.shape[0] == 1:
            return self.iou(x[0], y[0])
        elif x.dim() == 4:
            iou = torch.stack([self.forward(x[i], y[i]) for i in range(x.shape[0])])
            return iou
        elif x.dim() > 4:
            raise WrongTensorShapeException("Tensor shape must have at most 4 dimensions.")
        
        ious = []
        intersec = 0
        union = 0
        for i in range(x.shape[0]):
            ious.append(self.iou(x[i], y[i]))
            intersec = intersec + self.iou.intersec
            union = union + self.iou.union

        iou = torch.stack(ious)
        mean_iou = iou.mean(dim=-1, keepdim=True)
        tot_iou = intersec / union
        tot_iou = tot_iou.unsqueeze(0)
        iou = torch.cat((mean_iou, tot_iou, iou), dim=-1)
        return iou 


if __name__ == "__main__":
    x = torch.ones((3, 3, 3, 3))
    y = torch.ones((3, 3, 3, 3))
    iou = IoU()
    print(torch.mean(iou(x, y), dim=-1))
