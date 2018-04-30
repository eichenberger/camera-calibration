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
    mapfile = sys.argv[1]

    with io.open(mapfile, 'r') as f:
        pointmap = json.JSONDecoder().decode(f.read())

    tmp_points3d = list(map(lambda point: point['position'], pointmap))
    points3d = np.array(tmp_points3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, point3d in enumerate(points3d):
        ax.scatter(point3d[0], point3d[1], point3d[2],
                   marker='o')

    plt.show()

if __name__ == '__main__':
    main()
