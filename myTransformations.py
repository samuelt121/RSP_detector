import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
import cv2


# My transformations
class MyRandomRotation(object):
    """Rotate by one of the given angles."""

    def __init__(self, angle, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, image, bboxes):
        new_bboxes = torch.clone(bboxes)
        if random.random() < self.p:
            angle = random.randint(-self.angle, self.angle)
            image = TF.rotate(image, angle)

            bboxes_np = bboxes.numpy()
            w, h = image.shape[2], image.shape[1]
            cx, cy = w//2, h//2
            bboxes_coord = self.get_corners(bboxes_np)
            bboxes_coord_n = self.rotate_box(bboxes_coord, angle, cx, cy)
            new_bboxes = torch.tensor(self.create_bbox_Coordinates(bboxes_coord_n))

        return image, new_bboxes

    def get_corners(self, bboxes):
        return [[bboxes[0], bboxes[1]], [bboxes[0], bboxes[3]], [bboxes[2], bboxes[1]], [bboxes[2], bboxes[3]]]

    def rotate_box(self, corners, angle, cx, cy):
        corners = np.array(corners).reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
        Rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)

        return np.dot(Rotation_matrix, np.array(corners).T).T.astype(np.int64)

    #
    def create_bbox_Coordinates(self, rotated_points):
        rotated_points = rotated_points.clip(min=0)
        xmin, ymin = np.min(rotated_points, axis=0)
        xmax, ymax = np.max(rotated_points, axis=0)
        return [xmin, ymin, xmax, ymax]

    # x = np.array([point[0] for point in rotated_points])
    # y = np.array([point[0] for point in rotated_points])
    # x = x.clip(min=0, max=w)
    # y = y.clip(min=0, max=h)
    #
    # xmin = np.min(x)
    # ymin = np.min(y)
    # xmax = np.max(x)
    # ymax = np.max(y)


class MyRandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        new_bboxes = torch.clone(bboxes)

        if random.random() < self.p:
            img = TF.hflip(img)
            # Flip bbox coordinates
            new_bboxes[0] = img.size(2) - bboxes[2]
            new_bboxes[2] = img.size(2) - bboxes[0]

        return img, new_bboxes


# Torch transformations:

class ToTensor:

    def __call__(self, pic):
        return TF.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ConvertImageDtype(torch.nn.Module):

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image):
        return TF.convert_image_dtype(image, self.dtype)


class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.bbox_transformations = ['MyRandomHorizontalFlip', 'MyRandomRotation']

    def __call__(self, img, bboxes):
        for t in self.transforms:
            if t.__class__.__name__ in self.bbox_transformations:
                img, bboxes = t(img, bboxes)
            else:
                img = t(img)
        return img, bboxes

if __name__ == '__main__':
    pass


