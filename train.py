import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import config
from MyDataset import RSP_train_Dataset
from utils import load_checkpoint, save_checkpoint, predict

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Data
train_dataset = RSP_train_Dataset(root='RSP_images', csv_file='train_images_detection.csv',
                                  transforms=config.transformations)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # RSP + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
# print(model)

# Loss and Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(params, lr=0.005, betas=(0.5, 0.99), weight_decay=0.0005)

# Load pre-trained model
load_model = True

try:
    if load_model:
        load_checkpoint(torch.load(config.saved_model_name_detection), model, optimizer)
except FileNotFoundError:
    print("Model file doesn't exist")

loss_value = 0
# Train network
for epoch in range(config.NUM_EPOCHS):
    losses_values = []

    if epoch % int((config.NUM_EPOCHS-1)/2) == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        save_checkpoint(checkpoint, filename=config.saved_model_name_detection)
    for batch_idx, (image, target) in enumerate(train_loader):
        image = image.to(device)
        for key, value in target.items():
            target[key] = target[key].to(device)

        # forward
        loss_dict = model(image, [target])
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses_values.append(loss_value)

        # backward
        optimizer.zero_grad()
        losses.backward()

        optimizer.step()

    # print loss
    print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, loss:{loss_value}")

classes = ["Rock", "Scissors", "Paper"]

# Verify performance on test set
print("Checking accuracy on Training Set")
predict(train_loader, model, classes, isTrain=True)

