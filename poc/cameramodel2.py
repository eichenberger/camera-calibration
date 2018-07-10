import logging
import numpy as np
from math import sin, cos, floor, sqrt

log = logging.getLogger()

class CameraModel:
    """The sample camera model to use"""
    def __init__(self, resolution):
        self._resolution = resolution
        self._extrinsic_mat = [[0]*4]*3
        self._intrinsic_mat = [[0]*3]*3

    def _distortion(self, point):
        newp = [0,0,0]

        f = [self._intrinsic_mat[0,0], self._intrinsic_mat[1,1]]
        c = [self._intrinsic_mat[0,2], self._intrinsic_mat[1,2]]

        # Get point coordinates from center of image
        point_center = point[0:2] - c
        x = point_center[0] / f[0]
        y = point_center[1] / f[1]
        u = x
        v = y

        # We need the radius of the point from the principal component

        r = sqrt(x**2 + y**2)

        # Radial distortion
        newp[0] = u * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)
        newp[1] = v * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)

        # Tangential distortion
        newp[0] = newp[0] + 2*self._p[0]*u*v + self._p[1]*(r**2+2*u**2)
        newp[1] = newp[1] + 2*self._p[1]*u*v + self._p[0]*(r**2+2*v**2)

        # Calculate pixels again
        newp[0] = f[0]*newp[0]
        newp[1] = f[1]*newp[1]

        point = point + newp

        return point

    def set_intrinsic(self, intrinsic):
        self._intrinsic_mat = intrinsic

    def set_extrinsic(self, extrinsic):
        self._extrinsic_mat = extrinsic

    def update_point_cloud(self, point_cloud):
        n = point_cloud.shape[0]
        cam_mat = np.matmul(self._intrinsic_mat, self._extrinsic_mat)
        # Add ones to the 3d dimension (for translation)
        points3d = np.concatenate((point_cloud, np.ones((n,1))), axis=1)
        points2d = np.asarray(np.matmul(cam_mat, np.transpose(points3d)))
        # normalize third dimension to 1
        points2d = np.mat(points2d.transpose())
        points2d = np.asarray(points2d/points2d[:, 2])
        self.points2d = np.array(list(map(lambda point: self._distortion(point), points2d)))

    def get_image(self):
        image = np.zeros((self._resolution[0], self._resolution[1]))
        """ Get an image of a point cloud """
        for point in self.points2d:
            if point[0] < self._resolution[0] and \
                    point[1] < self._resolution[1] and\
                    point[0] >= 0 and point[1] >= 0:
                x = floor(point[0])
                y = floor(point[1])
                image[x, y] = 1

        return image

    def set_distortion(self, k, p):
        """ Add distortion to the camera model """
        self._k = k
        self._p = p


