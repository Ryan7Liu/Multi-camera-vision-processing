import numpy as np
import cv2

import time
import multiprocessing as mp
import pyperclip as pc
import torch
import os
import segmentation_models_pytorch as smp
import albumentations as albu
import matplotlib.pyplot as plt
from Minirec import anglerec

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_multi_camera():
    camera_ip_l = [
        "E:\Experiments/test_2022_12_8/2022-11-29-02-09-00.mp4"]  # input serial numbers of the camera
    print(camera_ip_l)
    cap = cv2.VideoCapture(camera_ip_l[0])
    # cap.set(6, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    # cap.set(propId=3, value=1280)  # set the width of the video
    # cap.set(propId=4, value=480)  # set the height of the video

    if cap.isOpened():
        print("yes")
    else:
        print("no")

    window_name = "real-time"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    record = False
    # ENCODER = 'se_resnext50_32x4d'
    # ENCODER_WEIGHTS = 'imagenet'
    # # CLASSES = ['Fiber']
    # # ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    # DEVICE = 'cuda'
    # best_model = torch.load('./best_model.pth')
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    while True:
        frame_0 = cap.read()[1]
        frame = cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB)
        # cv2.imshow(window_name, frame)
        transformed = get_preprocessing(preprocessing_fn)(image=frame)["image"]
        x_tensor = torch.from_numpy(transformed).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        # print(type(pr_mask))
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        # print(type(pr_mask))
        pr_maskg = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGR)
        # print(type(pr_maskg))
        imgangle = anglerec(frame_0, pr_maskg)
        # print(imgangle, type(imgangle))
        # imgs = [frame, imgangle]
        # imgss = np.concatenate(imgs, axis=0)
        cv2.imshow(window_name, imgangle)
        k = cv2.waitKey(1) & 0xff
        # visualize(
        #     predicted_mask=pr_mask
        # )
        if k == ord("r"):
            # out.release()
            now = time.strftime("%Y_%m_%d_%H_%M_%S_",
                                time.localtime(time.time()))  # add the time tag into the file name.
            path = "D:/test/" + now + window_name + ".avi"
            print("Recording " + path)
            out = cv2.VideoWriter(path, fourcc, 24.0, (1280, 480), True)
            record = True
        elif k == ord("n"):
            print("No recording " + path)
            out.release()
            record = False
        if record:
            out.write(imgangle)
        if k == ord('q'):
            print("Quit capturing")
            # The video stream is paused until pressing s button
            # Then turn to the most recent frame
            break
        if k == ord('s'):
            now_0 = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
            # add the time tag into the file name.
            pc.copy(now_0 + window_name + ".png")
            path_0 = "D:/test/image/" + now_0 + window_name + ".png"
            print("Saving image " + path_0)
            # pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(path_0, imgangle)
    out.release()
    cv2.destroyAllWindows()
    print("Ceasing")
    return None


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Perform image pre-processing operations

    Args:
        preprocessing_fn (callbale): Functions for data normalization
            (for each pre-trained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.PadIfNeeded(384, 480),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def visualize(**images): # "**"arbitrary keyword arguments, sequence not matter; "*"arbitrary arguments, sequence matters
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['Fiber']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    best_model = torch.load('./best_model.pth')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # run_multi_camera_in_a_window()
    run_multi_camera()
