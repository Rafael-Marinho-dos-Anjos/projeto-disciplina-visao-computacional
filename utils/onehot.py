import cv2
import numpy as np
from torch import from_numpy, Tensor


def one_hot_mask(image_path, colors) -> Tensor:
    """ Convert a color mask to a one-hot encoded tensor.

    :param image_path: Image path
    :type image_path: str
    :param colors: List with mask colors
    :type colort: list[tuple]

    :return: Tensor
    """
    image = cv2.imread(image_path)
    image = cv2.split(image)[2]

    output = np.zeros((len(colors), image.shape[0], image.shape[1]), dtype=np.float32)

    for i, color in enumerate(colors):
        mask = cv2.inRange(image, color[0], color[0])
        output[i][mask != 0] = 1

    output_tensor = from_numpy(output)

    return output_tensor


if __name__ == "__main__":
    # Function usage example

    colors = [
                (60, 16, 152),
                (132, 41, 246),
                (110, 193, 228),
                (254, 221, 58),
                (226, 169, 41),
                (155, 155, 155)
            ]
    print(one_hot_mask(r"dataset\Semantic segmentation dataset\Tile 1\masks\image_part_001.png", colors).max())
