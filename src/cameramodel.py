import logging
import numpy as np
from math import sin, cos, floor, sqrt

log = logging.getLogger()

class CameraModel:
    """The sample camera model to use"""
    def __init__(self, resolution, f = [20,20], c = None):
        if c == None:
            c = np.array(resolution)/2
        self._extrinsic_mat = np.mat([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self._resolution = resolution
        self._k = [0]*3
        self._p = [0]*2

        self._f = f
        self._c = c
        self._update_intrinsic()

    @staticmethod
    def _rot_mat_x(angle):
        return np.mat([[1, 0, 0],
                       [0, cos(angle), -sin(angle)],
                       [0, sin(angle), cos(angle)]])

    @staticmethod
    def _rot_mat_y(angle):
        return np.mat([[cos(angle), 0, sin(angle)],
                       [0, 1, 0],
                       [-sin(angle), 0, cos(angle)]])

    @staticmethod
    def _rot_mat_z(angle):
        return np.mat([[cos(angle), -sin(angle), 0],
                       [sin(angle), cos(angle), 0],
                       [0, 0, 1]])

    def _update_intrinsic(self):
        self._intrinsic_mat = np.mat([[self._f[0], 0, self._c[0]],
                                      [0, self._f[1], self._c[1]],
                                      [0, 0, 1]])

    def set_f(self, f):
        self._f = f
        self._update_intrinsic()

    def set_c(self, c):
        self._c = c
        self._update_intrinsic()

    def create_extrinsic(self, pose, translation):
        rot_x = self._rot_mat_x(pose[0])
        rot_y = self._rot_mat_y(pose[1])
        rot_z = self._rot_mat_z(pose[2])

        rot = np.matmul(rot_x, rot_y)
        rot = np.matmul(rot, rot_z)

        translation = np.mat(translation).transpose()
        self._extrinsic_mat = np.concatenate((rot, translation), axis=1)

    def _distortion(self, point):
        newp = [0,0,0]

        # Get point coordinates from center of image
        point_center = point[0:2] - self._c
        u = point_center[0] / self._f[0]
        v = point_center[1] / self._f[1]

        # We need the radius of the point from the principal component
        r = sqrt(u**2 + v**2)

        # Radial distortion
        newp[0] = u * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)
        newp[1] = v * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)

        # Tangential distortion
        newp[0] = newp[0] + 2*self._p[0]*u*v + self._p[1]*(r**2+2*u**2)
        newp[1] = newp[1] + 2*self._p[1]*u*v + self._p[0]*(r**2+2*v**2)

        # Calculate pixels again
        newp[0] = newp[0] * self._f[0]
        newp[1] = newp[1] * self._f[1]

        point = point + newp

        return point


    def update_point_cloud(self, point_cloud):
        """ Get an image of a point cloud """
        log.debug("Intrinsic:")
        log.debug(self._intrinsic_mat)
        log.debug("Extrinsic:")
        log.debug(self._extrinsic_mat)
        cam_mat = np.matmul(self._intrinsic_mat, self._extrinsic_mat)
        log.debug("C:")
        log.debug(cam_mat)
        self.points2d = np.zeros((len(point_cloud), 3))
        for i, point in enumerate(point_cloud):
            wp = np.ones((4))
            wp[0:3] = point[0:3]
            log.debug("WP:")
            log.debug(wp)
            impoint = np.matmul(cam_mat, wp)
            impoint = np.asarray(impoint)[0]
            if impoint[2] > 0:
                impoint = impoint/impoint[2] # normalize so that z is 1
                log.debug("Impoint:")
                log.debug(impoint)
                self.points2d[i, :] = self._distortion(impoint)

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

    def add_distortion(self, k, p):
        """ Add distortion to the camera model """
        self._k = k
        self._p = p


