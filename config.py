import torchvision.transforms as transforms
import torch
from cv2 import FONT_HERSHEY_SIMPLEX
from myTransformations import MyRandomRotation, MyRandomHorizontalFlip, MyCompose

Detection_threshold = 0.1

N_channels = 3
IMAGE_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 3e-5
NUM_EPOCHS = 100
NUM_CLASSES = 3

window_color = (0, 255, 0)
window_thickness = 2
text_font = FONT_HERSHEY_SIMPLEX
text_color = (255, 0, 0)
text_thickness = 2
text_scale = 1


transformations = MyCompose([
        transforms.ToPILImage(),
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.2),  # Change brightness of image
        transforms.ToTensor(),
        MyRandomHorizontalFlip(p=0.5),  # Flip image horizontally
        MyRandomRotation(75, p=1),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.ConvertImageDtype(torch.float32),
        # transforms.Lambda(lambda image: image.permute(1, 2, 0)),  # to get: (H,W,Ch)
    ])


testTransforms = transforms.Compose([
    transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
    transforms.ConvertImageDtype(torch.float32),
    ])

saved_model_name_classification = 'RSP_checkpoint_classification.pth.tar'
saved_model_name_detection = 'RSP_checkpoint_detection.pth.tar'

# https://github.com/pytorch/vision/issues/230
