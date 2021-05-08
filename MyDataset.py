import torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset  # Gives easier dataset managment and creates mini batches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from myTransformations import MyRandomRotation, MyRandomHorizontalFlip, MyCompose


class RSP_train_Dataset(Dataset):
    def __init__(self, root, csv_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]), dtype=torch.int64)
        bbox = (self.annotations.iloc[index, 2])[1:-1].split(",")
        bbox = [int(x) for x in bbox]
        boxes = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        boxes = torch.as_tensor(boxes, dtype=torch.int64)

        image_id = torch.tensor([index])
        area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)  # each image contains one class. non crowd units.

        target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target["boxes"] = self.transforms(image, boxes)

        return image, target


class RSP_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transforms:
            image = self.transforms(image)

        return image, y_label


# Test code
if __name__ == '__main__':
    trans = MyCompose([
        transforms.ToPILImage(),
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.2),  # Change brightness of image
        transforms.ToTensor(),
        MyRandomHorizontalFlip(p=0.5),  # Flip image horizontally
        MyRandomRotation(30, p=1),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(lambda image: image.permute(1, 2, 0)),  # to get: (H,W,Ch)
    ])

    dataset = RSP_train_Dataset(csv_file='train_images_detection.csv', root='RSP_images',
                                transforms=trans)

    cnt = 0
    for im, target in dataset:

        box = target['boxes'].detach().numpy()
        rect = Rectangle((int(box[0]), box[1]), box[2] - box[0], box[3] - box[1], edgecolor='r', facecolor='none')
        print(target['labels'])

        figure, ax = plt.subplots(1)
        ax.imshow(im)
        ax.add_patch(rect)
        figure.show()

        cnt = cnt + 1

    print('set debugging point')
