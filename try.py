import numpy as np

h = 0.5
w = 2
l = 3

x=0.1
y=0.2
z=0.3
corner_3d = [[l/2, h,  w/2],
                    [-l/2, h,  w/2],
                    [-l/2, h, -w/2],
                    [ l/2, h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]]

corner_3d = np.array(corner_3d)
corner_3d = np.hstack([corner_3d, np.ones((8,1))])
corner_3d = corner_3d.T

ry = 3.14/4
cos_ry = np.cos(ry)
sin_ry = np.sin(ry)
T = [[ cos_ry, 0, sin_ry, -x],
    [0,       1, 0,      -y],
    [-sin_ry, 0, cos_ry, -z],
    [0,       0, 0,      1]]
T = np.array(T)

corner_3d = np.dot(T, corner_3d)
#print(corner_3d)

corner_3d = corner_3d.T

corner_3d = corner_3d[:,:3]

print(corner_3d)