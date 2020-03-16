

import numpy as np
import cv2
import cv2.aruco as aruco


def test_1(marker, num):
    """
    marker = aruco.DICT_4X4_50
        4x4 bits, et 50 marker dans le dict

    Dictionary_get(marker)
        retourne le dict du marker

    drawMarker(aruco_dict, num, 700)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
        second parameter is id number
        last parameter is total image size
    """

    aruco_dict = aruco.Dictionary_get(marker)

    img = aruco.drawMarker(aruco_dict, num, 700)
    # #cv2.imwrite("test_marker.jpg", img)

    cv2.imshow('frame',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def test_2():
    markers_1 = [ aruco.DICT_4X4_50,
                  aruco.DICT_4X4_100,
                  aruco.DICT_4X4_250,
                  aruco.DICT_4X4_1000,]

    for m in markers_1:
        try:
            test_1(m, 5)
            print(m,"ok")
        except:
            print(m,"impossible")
        try:
            test_1(m, 255)
            print(m,"ok")
        except:
            print(m,"impossible")

def test_3():

    markers = [   aruco.DICT_4X4_250,
                  aruco.DICT_5X5_250,
                  aruco.DICT_6X6_250,
                  aruco.DICT_7X7_250]

    for m in markers:
        test_1(m, 2)
        print(m)

    for i in range(12):
        test_1(aruco.DICT_4X4_250, i)
        print(i)

test_2()
test_3()
