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

QUOI = 1

def main():
    # Récup des datas de la pyboard
    # #raw_data = gl.multi.receive()
    try:
        raw_data = gl.multi.receive()
    except:
        raw_data = None

    if raw_data:
        data_to_var(raw_data)
        if QUOI == 1:
            set_rot()
            set_trans()
        else:
            set_rot_one()
            set_trans_one()

def set_rot():

    alpha = - gl.rvec[2]
    if alpha > 0.18:
        alpha = - alpha + 0.37
    # #alpha = 0

    beta = gl.rvec[0] - 2
    if beta < -3:
        beta += 4
    # #beta = 0

    gamma = gl.rvec[1] + 2
    if gamma > 3.8:
        gamma -= 4.2
    # #gamma = 0

    print("angle", round(alpha, 2), round(beta, 2), round(gamma, 2))

    # set objects orientation with alpha, beta, gamma in radians
    rot_en_euler = mathutils.Euler([alpha, beta-0.2, gamma+0.2])

    # apply to objects local orientation if ok
    gl.cube.localOrientation = rot_en_euler.to_matrix()


def set_trans():

    x = -gl.tvec[0]/5 + 2
    y = gl.tvec[2]/5 - 10
    z = -gl.tvec[1]/5 + 2
    # #print("position", round(x, 2), round(y, 2), round(z, 2))
    gl.cube.position = x, y, z


def set_rot_one():

    alpha = gl.rvec[0] - 2.8 - 0.32
    if alpha < -4.5:
        alpha += 4.5

    beta = gl.rvec[1]
    if beta < -3:
        beta += 4

    gamma = gl.rvec[2] - 0.35 -0.4
    if gamma < -0.3:
        gamma += 0.8

    print("angle", round(alpha, 2), round(beta, 2), round(gamma, 2))

    # set objects orientation with alpha, beta, gamma in radians
    rot_en_euler = mathutils.Euler([alpha, beta, gamma])

    # apply to objects local orientation if ok
    gl.cube.localOrientation = rot_en_euler.to_matrix()


def set_trans_one():

    x = -gl.tvec[0]/2 - 5
    y = gl.tvec[2]/2 - 6
    z = -gl.tvec[1]/2 + 9.5

    # #print("position", round(x, 2), round(y, 2), round(z, 2))
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
