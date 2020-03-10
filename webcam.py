#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import subprocess
from time import sleep
from pymultilame import MyConfig

"""
serge@PC56:~$ uvcdynctrl -d video0 -v -c
Listing available controls for device video0:
  Brightness
    ID      : 0x00000001,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 30 .. 255, step size: 1 ],
    Default : 133
  Contrast
    ID      : 0x00000002,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 10, step size: 1 ],
    Default : 5
  Saturation
    ID      : 0x00000004,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 200, step size: 1 ],
    Default : 83
  White Balance Temperature, Auto
    ID      : 0x00000009,
    Type    : Boolean,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 1, step size: 1 ],
    Default : 1
  Power Line Frequency
    ID      : 0x0000000d,
    Type    : Choice,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : { 'Disabled'[0], '50 Hz'[1], '60 Hz'[2] },
    Default : 2
  White Balance Temperature
    ID      : 0x00000008,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 2800 .. 10000, step size: 1 ],
    Default : 4500
  Sharpness
    ID      : 0x00000007,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 50, step size: 1 ],
    Default : 25
  Backlight Compensation
    ID      : 0x0000000c,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 10, step size: 1 ],
    Default : 0
  Exposure, Auto
    ID      : 0x0000000f,
    Type    : Choice,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : { 'Manual Mode'[1], 'Aperture Priority Mode'[3] },
    Default : 1
  Exposure (Absolute)
    ID      : 0x00000011,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 5 .. 20000, step size: 1 ],
    Default : 156
  Pan (Absolute)
    ID      : 0x0000001c,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ -201600 .. 201600, step size: 3600 ],
    Default : 0
  Tilt (Absolute)
    ID      : 0x0000001e,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ -201600 .. 201600, step size: 3600 ],
    Default : 0
  Focus (absolute)
    ID      : 0x00000015,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 40, step size: 1 ],
    Default : 0
  Focus, Auto
    ID      : 0x00000014,
    Type    : Boolean,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 1, step size: 1 ],
    Default : 0
  Zoom, Absolute
    ID      : 0x00000019,
    Type    : Dword,
    Flags   : { CAN_READ, CAN_WRITE },
    Values  : [ 0 .. 10, step size: 1 ],
    Default : 0
"""

"""
                     brightness 0x00980900 (int)    : min=30 max=255 step=1 default=133 value=50
                       contrast 0x00980901 (int)    : min=0 max=10 step=1 default=5 value=5
                     saturation 0x00980902 (int)    : min=0 max=200 step=1 default=83 value=100
 white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=0
      white_balance_temperature 0x0098091a (int)    : min=2800 max=10000 step=1 default=4500 value=10000
                      sharpness 0x0098091b (int)    : min=0 max=50 step=1 default=25 value=25
         backlight_compensation 0x0098091c (int)    : min=0 max=10 step=1 default=0 value=1
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=1 value=1
              exposure_absolute 0x009a0902 (int)    : min=5 max=20000 step=1 default=156 value=150
                   pan_absolute 0x009a0908 (int)    : min=-201600 max=201600 step=3600 default=0 value=0
                  tilt_absolute 0x009a0909 (int)    : min=-201600 max=201600 step=3600 default=0 value=0
                 focus_absolute 0x009a090a (int)    : min=0 max=40 step=1 default=0 value=0
                     focus_auto 0x009a090c (bool)   : default=0 value=0
                  zoom_absolute 0x009a090d (int)    : min=0 max=10 step=1 default=0 value=0
"""

CAM_PARAM = [
    ["Brightness", "brightness"],
    ["Contrast", "contrast"],
    ["Saturation", "saturation"],
    ["White Balance Temperature, Auto", "w_bal_temp_aut"],
    ["Power Line Frequency", "power_line_freq"],
    ["White Balance Temperature", "white_bal_temp"],
    ["Sharpness", "sharpness"],
    ["Backlight Compensation", "backlight_compensation"],
    ["Exposure, Auto", "exposure_auto"],
    ["Exposure (Absolute)", "exposure_absolute"],
    ["Pan (Absolute)", "pan"],
    ["Tilt (Absolute)", "tilt"],
    ["Focus (absolute)", "focus_absolute"],
    ["Focus, Auto", "focus_auto"],
    ["Zoom, Absolute", "zoom_absolute"],
    ]


def apply_all_cam_settings(cf, cam):
    """Apply all settings on cam."""

    # List of all cam settings, disable auto settings first
    # An item is a tuple, ("uvcdynctrl parameter name", "variable in scan.ini")
    for item in CAM_PARAM:
        subprocess.call('uvcdynctrl -d video"{0}" -s  "{1}" {2}'.format(
                            cam, item[0], cf[item[1]]), shell=True)
        print("Cam{0} settings: {1} = {2}".format(cam, item[0], cf[item[1]]))
        sleep(0.1)

def apply_cam_setting(cam, name, value):
    """Apply only one setting on cam.
    ["Brightness", "brightness"]
    avec "brightness", j'applique "Brightness"
    """
    for item in CAM_PARAM:
        if item[1] == name:
            correct_name = item[0]

    subprocess.call('uvcdynctrl -d video"{0}" -s  "{1}" {2}'.format(cam,
                                                correct_name, value), shell=True)
    print("Cam{0} settings: {1} = {2}".format(cam, correct_name, value))


if __name__ == '__main__':
    dossier = os.getcwd()
    cf = MyConfig(dossier + "/aruco.ini")
    conf = cf.conf
    #subprocess.call("uvcdynctrl -c video0 -l", shell=True)
    apply_all_cam_settings(conf["HD5000"], cam=0)

    #apply_cam_setting(1, "Brightness", 33)
