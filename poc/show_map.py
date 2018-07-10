import sys
import io
import json

import numpy as np
import vispy.scene
from vispy.scene import visuals

import cv2
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

# Workaround so that orb compute doesn't crash
cv2.ocl.setUseOpenCL(False)


theta, phi = 0,0

def main():
    imagefile = sys.argv[1]
    depthimagefile = sys.argv[2]
    mapfile = sys.argv[3]

    image = cv2.imread(imagefile,cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_RGB2GRAY)
    # Initiate ORB detector
    orb = cv2.ORB_create(500, 1.4, 8, fastThreshold=20)
    # find the keypoints with ORB
    kp = orb.detect(gray, None)
    # compute the descriptors with ORB
    kps, image_descriptors = orb.compute(gray, kp)
    cv2.imshow("test", image)

    depth_image = cv2.imread(depthimagefile,cv2.IMREAD_UNCHANGED)/10

    with io.open(mapfile, 'r') as f:
        pointmap = json.JSONDecoder().decode(f.read())

    tmp_descriptors = list(map(lambda point: point['descriptors'], pointmap))
    map_descriptors = np.array(tmp_descriptors, dtype=np.uint8)

    tmp_points3d = list(map(lambda point: point['position'], pointmap))
    points3d = np.array(tmp_points3d)

    # match the orb points with the points from the map
    image_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    image_matches = image_matcher.match(image_descriptors, map_descriptors)

    out_image = image[:,:,0:3].copy()
#    colors = np.random.random((n, 3)) * 255
#    colors = colors.astype(int)
#    out_image = image[:,:,0:3].copy()
#    for i, kp in enumerate(image_kps_subset):
#        thecolor = color=colors[i,:]
#        out_image = cv2.drawKeypoints(out_image, [kp], image, color=thecolor.tolist())

    out_image = cv2.drawKeypoints(image, kps, out_image)
    image = out_image
#    indexes = list(map(lambda match: match.trainIdx, sorted_matches))
#    points3d_subset = points3d[indexes[0:n]]

#    colors = colors.astype(float)/255
#    for i in range(0, image.shape[0]):
#        for j in range(0, image.shape[1]):
#            depth = depth_image[i,j,3] if depth_image[i,j,3] != 0 else 0.1

    x_line = np.arange(image.shape[1])
    y_line = np.arange(image.shape[0])
    x = np.asarray(np.repeat(np.mat(x_line), image.shape[0], axis=0).flatten())[0]
    y = np.asarray(np.repeat(np.mat(y_line).transpose(), image.shape[1], axis=1).flatten())[0]
    depth = depth_image.flatten()
    color = image[:,:,0:3]/255
    color = color.reshape(len(y),3)

    pos = np.asarray([x,y,depth]).transpose()

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color=None, face_color=color, size=5)

    view.add(scatter)

    view.camera = 'arcball'  # or try 'arcball'

    axis = visuals.XYZAxis(parent=view.scene)

    vispy.app.run()

if __name__ == '__main__':
    main()
