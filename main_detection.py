import cv2, time
from utils import load_checkpoint, winner_is
import torch
import config
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # RSP + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
load_checkpoint(torch.load(config.saved_model_name_detection), model, optimizer)

# Set Windows
Video = cv2.VideoCapture(0)
Video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Detection_Window', flags=cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Result_Window', flags=cv2.WINDOW_KEEPRATIO)

# set Classes
classes = ["Rock", "Scissors", "Paper"]

rock = cv2.imread(r'RSP_images/R_rock.jpg')
scissors = cv2.imread(r'RSP_images/S_scissors.jpg')
paper = cv2.imread(r'RSP_images/M_paper.jpg')
blank = np.zeros_like(paper)

dictionary = {'Rock': rock, 'Scissors': scissors, 'Paper': paper}

# Start
model.eval()

# TODO: improve real time fps. check https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/.
# https://stackoverflow.com/questions/55828451/video-streaming-from-ip-camera-in-python-using-opencv-cv2-videocapture
while Video.isOpened():
    res, frame = Video.read()
    if not res:
        print("can't grab frame")
        break

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    frame_torch = config.testTransforms(frame)
    frame_torch = frame_torch.unsqueeze(0)
    frame_torch = frame_torch.to(device)
    with torch.no_grad():
        outputs = model(frame_torch)
        pred_classes = [classes[i - 1] for i in outputs[0]['labels'].cpu().numpy()]
        # get score for all the predicted objects
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        # get predicted bounding box
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        # get boxes above the threshold score
        if len(pred_bboxes) == 0 or len(pred_scores) == 0:
            continue
        box = pred_bboxes[0]

        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))

        if pred_scores[0] > config.Detection_threshold:
            pred_class = pred_classes[0]

            frame = cv2.rectangle(frame, start_point, end_point, config.window_color, config.window_thickness)
            frame = cv2.putText(frame, pred_class, start_point, config.text_font,
                                config.text_scale, config.text_color, config.text_thickness, cv2.LINE_AA)
            winner_image = dictionary[winner_is(pred_class)]
        else:
            winner_image = blank

        cv2.imshow('Result_Window', winner_image)
        cv2.imshow('Detection_Window', frame)
    time.sleep(1)
Video.release()

# Note: use 'pip install opencv-contrib-python --user' for solving opencv problem.