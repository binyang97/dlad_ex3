import numpy as np

def expand_bbox(box, delta=1.0):
    '''
    expand bounding box with given distance in all directions
    input (8, 3) corners coordinates
    '''

    direction = np.sign(box)

    direction[direction == 0] = 1
    
    newbox = box + direction * delta

    return newbox

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c
h = 0.5
w = 2
l = 3

x=0
y=0
z=0
corner_3d = [[l/2, -h,  w/2],
                    [-l/2, -h,  w/2],
                    [-l/2, -h, -w/2],
                    [ l/2, -h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]]

#corner_3d_expanded = [[l/2+1, -h-1,  w/2+1],
 #           [-l/2-1, -h-1,  w/2+1],
 #           [-l/2-1, -h-1, -w/2-1],
 #           [ l/2+1, -h-1, -w/2-1],
 #           [ l/2+1,  1,  w/2+1],
 #           [-l/2-1,  1,  w/2+1],
 #           [-l/2-1,  1, -w/2-1],
 #           [ l/2+1,  1, -w/2-1]]



corner_3d_expanded = expand_bbox(corner_3d)

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


#print(corner_3d)

print(box3d_vol(corner_3d))

print(box3d_vol(expand_bbox(corner_3d)))

corner_3d_expanded = np.array(corner_3d_expanded)
corner_3d_expanded = np.hstack([corner_3d_expanded , np.ones((8,1))])
corner_3d_expanded  = corner_3d_expanded .T

ry = 3.14/4
cos_ry = np.cos(ry)
sin_ry = np.sin(ry)
T = [[ cos_ry, 0, sin_ry, -x],
    [0,       1, 0,      -y],
    [-sin_ry, 0, cos_ry, -z],
    [0,       0, 0,      1]]
T = np.array(T)

corner_3d_expanded = np.dot(T, corner_3d_expanded )
#print(corner_3d)

corner_3d_expanded = corner_3d_expanded .T

corner_3d_expanded  = corner_3d_expanded [:,:3]

print(box3d_vol(corner_3d_expanded))
# mask_x = np.array([True, False, True])
# mask_y = np.array([False, False, False])

# a = np.array([1, 2, 3])
# mask = mask_x & mask_y
# print(mask)
# print(np.any(mask))

# #print(a[mask])