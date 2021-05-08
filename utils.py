import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Check accuracy on training to see how good our model is
def predict(loader, model, classes, isTrain):
    num_correct = 0
    num_samples = 0
    model.eval() #takes care of layers that behave differently during Evaluation mode. such as: BN, Dropout, etc.

    results = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device) # image
            if isTrain:
                label = y['labels'].to(device=device)
            else:
                label = y.to(device=device)

            outputs = model(x)

            pred_classes = [classes[i - 1] for i in outputs[0]['labels'].cpu().numpy()]
            # get score for all the predicted objects
            pred_scores = outputs[0]['scores'].detach().cpu().numpy()
            # get all the predicted bounding boxes
            pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
            # get boxes above the threshold score
            # boxes = pred_bboxes[pred_scores >= detection_th].astype(np.int32)
            box = pred_bboxes[0]
            pred_class = pred_classes[0]

            num_samples = num_samples + 1
            if classes[label - 1] == pred_classes[0]:
                num_correct = num_correct + 1

            results.append((box, pred_class))
            draw_box(x.squeeze(0).permute(1, 2, 0).cpu().numpy(), box, pred_class)
    print(f"{num_correct}/{num_samples} correct")


def draw_box(image, box, pred_class):
    # read the image with OpenCV

    figure, ax = plt.subplots(1)
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='r', facecolor='none')

    ax.imshow(image)
    ax.add_patch(rect)
    ax.text(box[0], box[1], pred_class)
    figure.show()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            for key, value in y.items():
                y[key] = y[key].to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


def winner_is(string):
    if string is "Paper":
        return "Scissors"
    elif string is "Rock":
        return "Paper"
    elif string is "Scissors":
        return "Rock"
