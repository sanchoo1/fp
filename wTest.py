import util
import numpy as np

rt = np.array([[    0.99465,    0.058428,     0.08516,      25.148],
[   0.055537,    -0.99781,    0.035933,     -7.6211],
[   0.087073,   -0.031011,    -0.99572,      539.24],
[          0,           0 ,          0 ,          1]])

dcam = np.array([[0], [0], [300], [1]])

w = util.getPosition(dcam, rt, util.MyKinect())

print(w)