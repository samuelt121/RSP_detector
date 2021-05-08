import cv2
from skimage import io
from glob import glob

# Naive implementation for labeling data.
# TODO: write results directly to file

images_path = r'C:\Users\User\PycharmProjects\RSP_solver\RSP_images'
images_names = glob(images_path + '/*')
images = [io.imread(img_name) for img_name in images_names]

ROIs = list()
images_classes = list()


for image in images:
    ROIs.append(cv2.selectROI(image))
    images_classes.append(input('The class is: '))


print(images_classes)
print(ROIs)

# ['M_paper', 'M_rock', 'M_scissors', 'R_paper', 'R_rock', 'R_scissors', 'S_paper', 'S_rock', 'S_scissors']
# [(128, 145, 332, 364), (161, 237, 226, 204), (142, 117, 266, 337), (464, 285, 194, 170), (504, 319, 110, 114), (478, 288, 129, 161), (277, 211, 176, 190), (307, 294, 123, 112), (275, 217, 157, 180)]