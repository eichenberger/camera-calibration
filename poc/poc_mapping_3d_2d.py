import sys
import io
import json

import cv2
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import vtk_visualizer

# Workaround so that orb compute doesn't crash
cv2.ocl.setUseOpenCL(False)


def main():
    imagefile = sys.argv[1]
    mapfile = sys.argv[2]

    image = cv2.imread(imagefile,cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_RGB2GRAY)
    # Initiate ORB detector
    orb = cv2.ORB_create(500, 1.4, 8, fastThreshold=20)
    # find the keypoints with ORB
    kp = orb.detect(gray, None)
    # compute the descriptors with ORB
    kps, image_descriptors = orb.compute(gray, kp)

    with io.open(mapfile, 'r') as f:
        pointmap = json.JSONDecoder().decode(f.read())

    tmp_descriptors = list(map(lambda point: point['descriptors'], pointmap))
    map_descriptors = np.array(tmp_descriptors, dtype=np.uint8)

    tmp_points3d = list(map(lambda point: point['position'], pointmap))
    points3d = np.array(tmp_points3d)

    # match the orb points with the points from the map
    image_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    image_matches = image_matcher.match(image_descriptors, map_descriptors)

    n = 20
    sorted_matches = sorted(image_matches, key=lambda match: match.distance)
    indexes = list(map(lambda match: match.queryIdx, sorted_matches))
    image_kps_subset = np.asarray(kps)[indexes[0:n]]

    colors = np.random.random((n, 3)) * 255
    colors = colors.astype(int)
    out_image = image[:,:,0:3].copy()
    for i, kp in enumerate(image_kps_subset):
        thecolor = color=colors[i,:]
        out_image = cv2.drawKeypoints(out_image, [kp], image, color=thecolor.tolist())

#    plt.imshow(out_image)

    indexes = list(map(lambda match: match.trainIdx, sorted_matches))
    points3d_subset = points3d[indexes[0:n]]

    depth_image = image[:, :, 3]
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    colors = colors.astype(float)/255
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            depth = image[i,j,3] if image[i,j,3] != 0 else 1

    x = np.asarray(np.repeat(np.mat(np.arange(image.shape[0])), image.shape[1], axis=0).flatten())[0]
    y = np.asarray(np.repeat(np.mat(np.arange(image.shape[1])), image.shape[0], axis=0).flatten())[0]
    depth = image[:,:,3].flatten()
    color = image[:,:,0:3]/255
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    color = color.reshape(len(y),3)
#    mlab.points3d(x, y, depth, color=color)
#    mlab.show()
    vtk_visualizer.plotxyz([x,y,depth,color])
#    ax.scatter(x, y, depth,
#                marker='o', c = color)
#    plt.show()

if __name__ == '__main__':
    main()
