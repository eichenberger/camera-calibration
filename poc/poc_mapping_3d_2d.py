import sys
import io
import json

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Workaround so that orb compute doesn't crash
cv2.ocl.setUseOpenCL(False)


def main():
    imagefile = sys.argv[1]
    mapfile = sys.argv[2]

    image = cv2.imread(imagefile,0)
    # Initiate ORB detector
    orb = cv2.ORB_create(1000, 1.4, 8, fastThreshold=20)
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kps, image_descriptors = orb.compute(image, kp)

    with io.open(mapfile, 'r') as f:
        pointmap = json.JSONDecoder().decode(f.read())

    tmp_descriptors = list(map(lambda point: point['descriptors'], pointmap))
    map_descriptors = np.array(tmp_descriptors, dtype=np.uint8)

    tmp_points3d = list(map(lambda point: point['position'], pointmap))
    points3d = np.array(tmp_points3d)

    # match the orb points with the points from the map
    image_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    image_matches = image_matcher.match(image_descriptors, map_descriptors)

    n = 10
    sorted_matches = sorted(image_matches, key=lambda match: match.distance)
    indexes = list(map(lambda match: match.queryIdx, sorted_matches))
    image_kps_subset = np.asarray(kps)[indexes[0:n]]

    colors = np.random.random((n, 3)) * 255
    colors = colors.astype(int)
    out_image = image.copy()
    for i, kp in enumerate(image_kps_subset):
        thecolor = color=colors[i,:]
        out_image = cv2.drawKeypoints(out_image, [kp], image, color=thecolor.tolist())

    plt.imshow(out_image)

    indexes = list(map(lambda match: match.trainIdx, sorted_matches))
    points3d_subset = points3d[indexes[0:n]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = colors.astype(float)/255
#    ax.scatter(points3d_subset[:, 0], points3d_subset[:, 1], points3d_subset[:, 2],
#            marker='o')
    for i, point3d in enumerate(points3d_subset):
        ax.scatter(point3d[0], point3d[1], point3d[2],
                   marker='o', c = colors[i])

    plt.show()

if __name__ == '__main__':
    main()
