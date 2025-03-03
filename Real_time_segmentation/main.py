import numpy as np
import cv2

import time
import multiprocessing as mp
import pyperclip as pc
import torch
import os
import segmentation_models_pytorch as smp
import albumentations as albu
from Minirec import anglerec
import Serialexcel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ensure the input and output of the video
def image_put(q, ip):
    print(ip)
    cap = cv2.VideoCapture(ip)
    # cap.set(6, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    # cap.set(propId=3, value=1280)  # set the width of the video
    # cap.set(propId=4, value=480)  # set the height of the video

    if cap.isOpened():
        print("yes")
    else:
        print("no")
    while cap.isOpened():
        q.put(cap.read()[1])  # put the frame in the queue, cap,read()[1],is frame, cause  {ret, frame = cap.read()}
        q.get() if q.qsize() > 1 else time.sleep(0.01)
    cap.release()


def image_get(q, window_name):

    window_name = str(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    record = False
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['Fiber']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    best_model = torch.load('./best_model.pth')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    while True:
        frame = q.get()
        transformed = get_preprocessing(preprocessing_fn)(image=frame)["image"]
        x_tensor = torch.from_numpy(transformed).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        cv2.imshow(window_name, pr_mask)
        k = cv2.waitKey(1) & 0xff
        if k == ord("r"):
            # out.release()
            now = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))   # add the time tag into the file name.
            path = "D:/test/" + now + window_name + ".avi"
            print("Recording " + path)
            out = cv2.VideoWriter(path, fourcc, 24.0, (1280, 480), True)
            record = True
        elif k == ord("n"):
            print("No recording " + path)
            out.release()
            record = False
        if record:
            out.write(frame)
        if k == ord('q'):
            print("Quit capturing")
            # The video stream is paused until pressing s button
            # Then turn to the most recent frame
            break
        if k == ord('s'):
            now_0 = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
            # add the time tag into the file name.
            pc.copy(now_0 + window_name + ".jpg")
            path_0 = "D:/test/image/" + now_0 + window_name + ".jpg"
            print("Saving image " + path_0)
            cv2.imwrite(path_0, frame)
    cv2.destroyAllWindows()
    print("Ceasing")
    return None


# capture the frame and storage
def image_collect(queue_list, camera_ip_l):
    """show in single opencv-imshow window"""
    # window_name = "Multi-views"
    window_name = "%s_and_so_on" % camera_ip_l[0]
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    record = False
    # storage path
    while True:
        imgs = [q.get() for q in queue_list]
        imgs = np.concatenate(imgs, axis=1)
        frame_0 = imgs
        record = False
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['Fiber']
        ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE = 'cuda'
        best_model = torch.load('./best_model.pth')
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
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
        if k == ord("r"):
            # out.release()
            now = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
            path = "D:/test/" + now + window_name + ".avi"
            print("Recording " + path)
            out = cv2.VideoWriter(path, fourcc, 24.0, (1280, 480), True)
            record = True
        elif k & 0xff == ord("n"):
            print("No recording " + path)
            out.release()
            record = False
        if record:
            out.write(imgs)
        if k == ord('q'):
            print("Quit capturing")
            # The video stream is paused until pressing s button
            # Then turn to the most recent frame
            break
        if k == ord('s'):
            now_0 = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
            # add the time tag into the file name.
            pc.copy(now_0 + window_name + ".jpg")
            path_0 = "D:/test/image/" + now_0 + window_name + ".jpg"
            print("Saving image " + path_0)
            cv2.imwrite(path_0, imgs)


def run_multi_camera():
    camera_ip_l = [
        "E:\Experiments/2022_5_30_Experiments_sample_1&3/2022_05_30_23_18_53_1_and_so_on.avi"]  # input serial numbers of the camera
    mp.set_start_method(method='spawn')  # initiate the multiple processing
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]  # creat a list of queues, 2 queues totally in this case
    # In each queue, the max element number are 4.
    # processes = [mp.Process(target=image_get, args=(queues, camera_ip_l))]
    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):  # creat multiprocessing, 4 threads in this case
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))
        processes.append(mp.Process(target=image_put, args=(queue, camera_ip)))
    for i in range(len(processes)):
        processes[i].daemon = True  # setattr(process, 'daemon', True)
        processes[i].start()
    # for process in processes:
    processes[1].join()
    # processes[3].join()
    if processes[1] is None:
        print("1")
        processes[0].terminate()
    if processes[3] is None:
        print("3")
        processes[2].terminate()


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def run_multi_camera_in_a_window():
    camera_ip_l = [0, 1, 2]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = [mp.Process(target=image_collect, args=(queues, camera_ip_l))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['Fiber']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    best_model = torch.load('./best_model.pth')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    run_multi_camera_in_a_window()
    # run_multi_camera()
