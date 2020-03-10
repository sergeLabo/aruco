#!python3
# -*- coding: UTF-8 -*-


"""


This code assumes that images used for calibration are of the same arUco marker
board provided with code

Mes images sont dans /cam_data
"""


import time
import cv2
import cv2.aruco as aruco
import numpy as np
from webcam import apply_all_cam_settings, apply_cam_setting
import yaml
from pathlib import Path
from tqdm import tqdm

from myconfig import MyConfig

class WebcamSettings:

    def __init__(self, cam, cf):

        self.cam = cam
        self.cap = cv2.VideoCapture(self.cam)

        # L'objet config de aruco.ini
        self.cf = cf

        self.brightness = self.cf.conf['HD5000']['brightness']
        self.contrast = self.cf.conf['HD5000']['contrast']
        self.saturation = self.cf.conf['HD5000']['saturation']

        self.w_bal_temp_aut = self.cf.conf['HD5000']['w_bal_temp_aut']
        self.power_line_freq = self.cf.conf['HD5000']['power_line_freq']

        self.exposure_auto = self.cf.conf['HD5000']['exposure_auto']
        self.exposure_absolute = self.cf.conf['HD5000']['exposure_absolute']

        self.white_bal_temp = self.cf.conf['HD5000']['white_bal_temp']
        self.backlight_compensation = self.cf.conf['HD5000']['backlight_compensation']

        self.sharpness = self.cf.conf['HD5000']['sharpness']
        self.pan = self.cf.conf['HD5000']['pan']
        self.tilt = self.cf.conf['HD5000']['tilt']
        self.focus_absolute = self.cf.conf['HD5000']['focus_absolute']
        self.focus_auto = self.cf.conf['HD5000']['focus_auto']
        self.zoom_absolute = self.cf.conf['HD5000']['zoom_absolute']

        # Trackbars
        self.create_trackbar()
        self.set_init_tackbar_position()

    def create_trackbar(self):
        """
        brightness (int): min=30 max=255 step=1 default=133 value=50
        contrast (int): min=0 max=10 step=1 default=5 value=5
        saturation (int): min=0 max=200 step=1 default=83 value=100
        white_balance_temperature_auto (bool): default=1 value=0
        power_line_frequency (menu) : min=0 max=2 default=2 value=0
        white_balance_temperature (int): min=2800 max=10000 step=1 default=4500 value=10000
        backlight_compensation (int): min=0 max=10 step=1 default=0 value=1
        exposure_auto (menu): min=0 max=3 default=1 value=1
        exposure_absolute (int): min=5 max=20000 step=1 default=156 value=150

        sharpness int): min=0 max=50 step=1 default=25 value=25
        pan_absolute (int): min=-201600 max=201600 step=3600 default=0 value=0
        tilt (int): min=-201600 max=201600 step=3600 default=0 value=0
        focus_absolute (int): min=0 max=40 step=1 default=0 value=0
        focus_auto (bool): default=0 value=0
        zoom_absolute (int): min=0 max=10 step=1 default=0 value=0
        """

        cv2.namedWindow('Reglage')
        self.reglage_img = np.zeros((10, 1400, 3), np.uint8)

        cv2.createTrackbar('brightness', 'Reglage', 0, 255, self.onChange_brightness)
        cv2.createTrackbar('contrast', 'Reglage', 0, 10, self.onChange_contrast)
        cv2.createTrackbar('saturation', 'Reglage', 0, 200, self.onChange_saturation)
        cv2.createTrackbar('w_bal_temp_aut', 'Reglage', 0, 1, self.onChange_w_bal_temp_aut)
        cv2.createTrackbar('power_line_freq', 'Reglage', 0, 2, self.onChange_power_line_freq)
        cv2.createTrackbar('white_bal_temp', 'Reglage', 2800, 10000, self.onChange_white_bal_temp)
        cv2.createTrackbar('backlight_compensation', 'Reglage', 0, 10, self.onChange_backlight_compensation)
        cv2.createTrackbar('exposure_auto', 'Reglage', 0, 3, self.onChange_exposure_auto)
        cv2.createTrackbar('exposure_absolute', 'Reglage', 5, 20000, self.onChange_exposure_absolute)

        cv2.createTrackbar('sharpness', 'Reglage', 0, 50, self.onChange_sharpness)
        cv2.createTrackbar('pan', 'Reglage', -201600, 201600, self.onChange_pan)
        cv2.createTrackbar('tilt', 'Reglage', -201600, 201600, self.onChange_tilt)
        cv2.createTrackbar('focus_absolute', 'Reglage', 0, 40, self.onChange_focus_absolute)
        cv2.createTrackbar('focus_auto', 'Reglage', 0, 1, self.onChange_focus_auto)
        cv2.createTrackbar('zoom_absolute', 'Reglage', 0, 10, self.onChange_zoom_absolute)

    def set_init_tackbar_position(self):
        """setTrackbarPos(trackbarname, winname, pos) -> None"""

        cv2.setTrackbarPos('brightness', 'Reglage', self.brightness)
        cv2.setTrackbarPos('saturation', 'Reglage', self.saturation)
        cv2.setTrackbarPos('exposure_auto', 'Reglage', self.exposure_auto)
        cv2.setTrackbarPos('exposure_absolute', 'Reglage', self.exposure_absolute)
        cv2.setTrackbarPos('contrast', 'Reglage', self.contrast)
        cv2.setTrackbarPos('w_bal_temp_aut', 'Reglage', self.w_bal_temp_aut)
        cv2.setTrackbarPos('power_line_freq', 'Reglage', self.power_line_freq)
        cv2.setTrackbarPos('white_bal_temp', 'Reglage', self.white_bal_temp)
        cv2.setTrackbarPos('backlight_compensation', 'Reglage', self.backlight_compensation)

        cv2.setTrackbarPos('sharpness', 'Reglage', self.sharpness)
        cv2.setTrackbarPos('pan', 'Reglage', self.pan)
        cv2.setTrackbarPos('tilt', 'Reglage', self.tilt)
        cv2.setTrackbarPos('focus_absolute', 'Reglage', self.focus_absolute)
        cv2.setTrackbarPos('focus_auto', 'Reglage', self.focus_auto)
        cv2.setTrackbarPos('zoom_absolute', 'Reglage', self.zoom_absolute)

    def onChange_brightness(self, brightness):
        """min=30 max=255 step=1 default=133
        """
        if brightness < 30: brightness = 30
        self.brightness = brightness
        self.save_change('HD5000', 'brightness', brightness)

    def onChange_saturation(self, saturation):
        """min=0 max=200 step=1 default=83
        """
        self.saturation = saturation
        self.save_change('HD5000', 'saturation', saturation)

    def onChange_exposure_auto(self, exposure_auto):
        """min=0 max=3 default=1
        """
        self.exposure_auto = exposure_auto
        self.save_change('HD5000', 'exposure_auto', exposure_auto)

    def onChange_exposure_absolute(self, exposure_absolute):
        """min=5 max=20000 step=1 default=156
        """
        self.exposure_absolute = exposure_absolute
        self.save_change('HD5000', 'exposure_absolute', exposure_absolute)

    def onChange_contrast(self, contrast):
        """min=0 max=10 step=1
        """
        self.contrast =contrast
        self.save_change('HD5000', 'contrast', contrast)

    def onChange_w_bal_temp_aut(self, w_bal_temp_aut):
        """min=0 max=1
        """
        self.w_bal_temp_aut = w_bal_temp_aut
        self.save_change('HD5000', 'w_bal_temp_aut', w_bal_temp_aut)

    def onChange_power_line_freq(self, power_line_freq):
        """min=0 max=2
        """
        self.power_line_freq = power_line_freq
        self.save_change('HD5000', 'power_line_freq', power_line_freq)

    def onChange_white_bal_temp(self, white_bal_temp):
        """white_bal_temp    min=2800 max=10000
        """
        if white_bal_temp < 2800: white_bal_temp = 2800
        self.white_bal_temp = white_bal_temp
        self.save_change('HD5000', 'white_bal_temp', white_bal_temp)

    def onChange_backlight_compensation(self, backlight_compensation):
        """min=0 max=10 step=1
        """
        self.backlight_compensation = backlight_compensation
        self.save_change('HD5000', 'backlight_compensation', backlight_compensation)

    def onChange_sharpness(self, sharpness):
        """sharpness int): min=0 max=50 step=1 default=25 value=25"""

        self.sharpness = sharpness
        self.save_change('HD5000', 'sharpness', sharpness)

    def onChange_pan(self, pan_absolute):
        """min=-201600 max=201600 step=3600 default=0 value=0"""

        self.pan = pan
        self.save_change('HD5000', 'pan', pan)

    def onChange_tilt(self, tilt):
        """min=-201600 max=201600 step=3600 default=0 value=0"""

        self.tilt = tilt
        self.save_change('HD5000', 'tilt', tilt)

    def onChange_focus_absolute(self, focus_absolute):
        """min=0 max=40 step=1 default=0 value=0"""

        self.focus_absolute = focus_absolute
        self.save_change('HD5000', 'focus_absolute', focus_absolute)

    def onChange_focus_auto(self, focus_auto):
        """default=0 value=0"""

        self.focus_auto = focus_auto
        self.save_change('HD5000', 'focus_auto', focus_auto)

    def onChange_zoom_absolute(self, zoom_absolute):
        """min=0 max=10 step=1 default=0 value=0"""

        self.zoom_absolute = zoom_absolute
        self.save_change('HD5000', 'zoom_absolute', zoom_absolute)

    def save_change(self, section, key, value):

        self.cf.save_config(section, key, value)
        if section == 'HD5000':
            apply_cam_setting(self.cam, key, value)


class ArucoCalibrateWebcam(WebcamSettings):

    def __init__(self, cam, cf):
        super().__init__(cam, cf)

        self.width = 1280
        self.height = 720
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        # For validating results, show aruco board to camera.
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
        self.aruco_params =  aruco.DetectorParameters_create()

        # Provide length of the marker's side, measurement unit is centimetre.
        self.marker_length = 3.80
        # Provide separation between markers, measurement unit is centimetre.
        self.marker_separation = 0.5

        # create arUco board
        self.board = aruco.GridBoard_create(4, 5,
                                            self.marker_length,
                                            self.marker_separation,
                                            self.aruco_dict)

        # finir le path avec un /
        self.path = "/media/data/3D/projets/aruco/cam_data/"
        self.count = 0
        self.loop = 1
        self.t = time.time()

    def get_data_calibration(self):
        """Generating data: 50 images
        Provide desired path to store images.
        Capture auto toutes les 2 secondes
        Esc to quit
        """

        while self.count < 50:
            name = self.path + str(self.count)+".jpg"
            ret, img = self.cap.read()

            cv2.imshow("Original", img)

            # Affichage des trackbars
            cv2.imshow('Reglage', self.reglage_img)
            print("ok",self.count)
            if time.time() - self.t > 2:
                cv2.imwrite(name, img)
                print("count", self.count, name)
                self.count += 1
                self.t = time.time()

            k = cv2.waitKey(10) & 0xFF
            if k == 27:  # ord('q'):
                break
        cv2.destroyAllWindows()

    def calibrate(self):
        """calibrating camera
        """

        # root directory of repo for relative path specification.
        root = Path(__file__).parent.absolute()

        # Set path to the images
        calib_imgs_path = root.joinpath("cam_data")

        # uncomment following block to draw and show the board
        img = self.board.draw((720, 1280))
        cv2.imshow("aruco", img)

        img_list = []
        calib_fnms = calib_imgs_path.glob('*.jpg')
        print('Using ...', end='')
        for idx, fn in enumerate(calib_fnms):
            print(idx, '', end='')
            img = cv2.imread(str(root.joinpath(fn)))
            img_list.append( img )
            h, w, c = img.shape
        print('Calibration images')

        counter, corners_list, id_list = [], [], []
        first = True
        for im in tqdm(img_list):
            img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                                                img_gray,
                                                self.aruco_dict,
                                                parameters=self.aruco_params)
            if first == True:
                corners_list = corners
                id_list = ids
                first = False
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
            counter.append(len(ids))
        print('Found {} unique markers'.format(np.unique(ids)))

        counter = np.array(counter)
        print ("Calibrating camera .... Please wait...")
        #mat = np.zeros((3,3), float)
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list,
                                                        id_list, counter,
                                                        self.board,
                                                        img_gray.shape, None, None )

        a = "Camera matrix is \n"
        b = "\n And is stored in calibration.yaml file\n"
        c = "along with distortion coefficients : \n"
        print(a, mtx, b+c, dist)
        data = {'camera_matrix': np.asarray(mtx).tolist(),
                'dist_coeff': np.asarray(dist).tolist()}
        with open("calibration.yaml", "w") as f:
            yaml.dump(data, f)

        cv2.destroyAllWindows()

    def validating_real_time_results(self):
        """
        image = cv.aruco.drawAxis(image, cameraMatrix, distCoeffs,
                                    rvec, tvec, length)
        Parameters
            image   input/output image. It must have 1 or 3 channels.
                    The number of channels is not altered.
            cameraMatrix    input 3x3 floating-point camera matrix
            distCoeffs  vector of distortion coefficients
            rvec    rotation vector of the coordinate system that will be drawn.
            tvec    translation vector of the coordinate system that will be drawn.
            length  length of the painted axis in the same unit than tvec
                    (usually in meters)
        """

        print("\n\nValidating real time results ...\n")
        with open('calibration.yaml') as f:
            loadeddict = yaml.load(f)
        mtx = loadeddict.get('camera_matrix')
        dist = loadeddict.get('dist_coeff')
        mtx = np.array(mtx)
        dist = np.array(dist)

        ret, img = self.cap.read()
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        h, w = img_gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                          dist,
                                                          (w,h),
                                                          1,
                                                          (w,h))

        pose_r, pose_t = [], []
        while True:
            ret, img = self.cap.read()
            img_aruco = img
            im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            h,  w = im_gray.shape[:2]
            dst = cv2.undistort(im_gray,
                                mtx,
                                dist,
                                None,
                                newcameramtx)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                                                dst,
                                                self.aruco_dict,
                                                parameters=self.aruco_params)

            if corners == None:
                print ("pass")
            else:
                rvec = np.zeros((3,), dtype=np.float32)
                tvec = np.zeros((3,), dtype=np.float32)

                # For a board
                ret, rvec, tvec = aruco.estimatePoseBoard(corners,
                                                          ids,
                                                          self.board,
                                                          newcameramtx,
                                                          dist,
                                                          rvec,
                                                          tvec,
                                                          useExtrinsicGuess=0)

                print("ret =", ret)
                print("rvec =", rvec)
                print("tvec =", tvec)
                print("Rotation ", rvec)
                print("Translation", tvec)

                if ret != 0:
                    img_aruco = aruco.drawDetectedMarkers(img,
                                                          corners,
                                                          ids,
                                                         (0,255,0))

                    # axis length 100 can be changed according to your requirement
                    img_aruco = aruco.drawAxis(img_aruco,
                                               newcameramtx,
                                               dist,
                                               rvec,
                                               tvec,
                                               10)

                    cv2.imshow("World co-ordinate frame axes", img_aruco)

                    # Send to Blender
                    self.sender.send([rvec, tvec])

                if cv2.waitKey(10) == 27:
                    break

        cv2.destroyAllWindows()



if __name__ == "__main__":

    cam = 0
    cf = MyConfig("./aruco.ini")
    conf = cf.conf
    apply_all_cam_settings(conf["HD5000"], cam)

    acw = ArucoCalibrateWebcam(cam, cf)
    # #acw.get_data_calibration()
    # #acw.calibrate()
    acw.validating_real_time_results()
