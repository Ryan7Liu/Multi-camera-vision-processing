import cv2
import numpy as np
from matplotlib import pyplot as plt


def anglerec(frame, mask): #
    font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.imread('test.png')
    imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # print(imgray, type(imgray))
    # cv2.imshow("imgray", imgray)
    # cv2.waitKey(0)
    ret, thresh = cv2.threshold(imgray, 0, 255, 1)
    # mask2 = mask.copy()
    # print(thresh, type(thresh))
    thresh = thresh.astype(np.uint8)
    # cv2.imshow("imgray", thresh)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # imgcon = cv2.drawContours(img2, contours, 1, (0, 0, 255), 3)
    # cv2.imshow("imgcon", imgcon)
    # cv2.waitKey(0)
    # cv2.imwrite("gray.png", thresh)
    # cnt = contours[1]

    for i in range(1, len(contours)):
        cnt = contours[i]
        # img2 = img.copy()
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # print(box)
        box = np.int0(box)  # 获得矩形角点
        # print(box)
        area = cv2.contourArea(box)
        width = rect[1][0]
        height = rect[1][1]
        angle = round(90 - rect[2], 2)
        cv2.polylines(frame, [box], True, (0, 128, 255), 3)
        # cv2.polylines(mask, [box], True, (0, 128, 255), 3)
        # text1 = 'Width: ' + str(int(width)) + ' Height: ' + str(int(height))
        # text2 = 'Rect Area: ' + str(area)
        text3 = 'Rect angle: ' + str(angle)
        # cv2.putText(img2, text1, (10+200*(i-1), 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
        # cv2.putText(img2, text2, (10+200*(i-1), 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
        cv2.putText(frame, text3, (10 + 200 * (i - 1), 30), font, 0.5, (0, 128, 255), 1, cv2.LINE_AA, 0)
        # cv2.putText(mask, text3, (10 + 200 * (i - 1), 30), font, 0.5, (0, 128, 255), 1, cv2.LINE_AA, 0)
    # cv2.imwrite('contours.png', img)
    # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Rectangle')
    # plt.imshow(img2)
    # cv2.imwrite('angle.png', img2)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # print(img2, type(img2))l
    return frame






if __name__ == "__main__":
    mask = cv2.imread('E:\Label_data_2022_6_21\data_masks\masks/2022_05_30_22_02_10_1_and_so_on.png')
    frame = cv2.imread('E:\Label_data_2022_6_21\data_masks\images/2022_05_30_22_02_10_1_and_so_on.jpg')
    # print(img, type(img))
    img2 = anglerec(frame, mask)
    cv2.imshow("imgangle", img2)
    cv2.waitKey(0)
#
# def cal(centercoord):  # determine the zone of the angle then calculate the direction
#     if centercoord[1] < 640: