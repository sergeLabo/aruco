#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

########################################################################
# This file is part of ArUCo.
#
# ArUCo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ArUCo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
########################################################################

'''
Lancé à chaque frame durant tout le jeu.
'''


from bge import logic as gl
import mathutils
import ast


def main():
    # Récup des datas de la pyboard
    # #raw_data = gl.multi.receive()
    try:
        raw_data = gl.multi.receive()
    except:
        raw_data = None

    if raw_data:
        data_to_var(raw_data)
        set_rot_trans()

"""
{'rvec': [-0.042519230395555496, -2.772643804550171, 0.22376395761966705],
'tvec': [17.984437942504883, -27.281139373779297, 116.52149200439453]}
"""


def set_rot_trans():

    alpha = gl.rvec[0]
    beta = gl.rvec[1]
    gamma = gl.rvec[2]

    # #print(round(alpha, 2), round(beta, 2), round(gamma, 2))

    # set objects orientation with alpha, beta, gamma in radians
    # #rot_en_euler_cam = mathutils.Euler([alpha, beta, gamma])
    rot_en_euler_cam = mathutils.Euler([alpha, beta, gamma])


    # apply to objects local orientation if ok
    gl.cube.localOrientation = rot_en_euler_cam.to_matrix()

    # #gl.plane.localScale = 1, 1, beta
    x = gl.tvec[0]/10
    y = gl.tvec[1]/10
    z = gl.tvec[2]/100
    print(round(x, 2), round(y, 2), round(z, 2))
    gl.cube.position = x, y, z

def datagram_to_dict(data):
    """Décode le message. Retourne un dict ou None."""

    try:
        dec = data.decode("utf-8")
    except:
        print("Décodage UTF-8 impossible")
        dec = data

    try:
        msg = ast.literal_eval(dec)
    except:
        print("ast.literal_eval impossible")
        print("Ajouter ast dans les import")
        msg = dec

    if isinstance(msg, dict):
        return msg
    else:
        print("Message reçu: None")
        return None


def data_to_var(data):
    data = datagram_to_dict(data)
    if data:
        if "rvec" in data:
            gl.rvec = data["rvec"]
    if data:
        if "tvec" in data:
            gl.tvec = data["tvec"]
