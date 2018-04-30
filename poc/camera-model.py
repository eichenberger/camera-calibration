"""
File: camera-model.py
Author: Stefan Eichenberger
Email: eichest@gmail.com
Github: eichenberger
Description: This is a proof of concept for point cloud base camera calibration
"""
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

#from pointmodel import PointModel
from pointmodel2 import PointModel2
from cameramodel import CameraModel

class Application:
    def __init__(self, model):
        self.resolution = [800, 800]
        self.fx = 300
        self.fy = 300
        self.cx = self.resolution[0] / 2
        self.cy = self.resolution[1] / 2
        self.thetax = 0
        self.thetay = 0
        self.thetaz = 0
        self.tx = -0.5
        self.ty = -0.5
        self.tz = 1
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.p1 = 0
        self.p2 = 0


        self.fx2 = 400
        self.fy2 = 400
        self.tz2 = 2

        self.model = model
        self.ax = plt.subplot(111)

    def show_image(self):
        image = self.update_image()*255 - self.update_image2()*255
        self.ax.imshow(image)

    def update_focus(self, val):
        self.fx = val
        self.fy = val
        self.show_image()

    def update_tz(self, val):
        self.tz = val
        self.show_image()

    def update_focus2(self, val):
        self.fx2 = val
        self.fy2 = val
        self.show_image()

    def update_tz2(self, val):
        self.tz2 = val
        self.show_image()

    def update_image(self):
        cm = CameraModel(self.resolution, [self.fx, self.fy], [self.cx, self.cy])
        cm.add_distortion([self.k1, self.k2, self.k3], [self.p1, self.p2])
        cm.create_extrinsic([self.thetax, self.thetay, self.thetaz], [self.tx, self.ty, self.tz])
        cm.update_point_cloud(self.model.points)

        return cm.get_image()

    def update_image2(self):
        cm = CameraModel(self.resolution, [self.fx2, self.fy2], [self.cx, self.cy])
        cm.add_distortion([self.k1, self.k2, self.k3], [self.p1, self.p2])
        cm.create_extrinsic([self.thetax, self.thetay, self.thetaz], [self.tx, self.ty, self.tz2])
        cm.update_point_cloud(self.model.points)

        return cm.get_image()

    def show(self):
        self.show_image()

        axfocus = plt.axes([0.25, 0.2, 0.65, 0.01], facecolor='b')
        axtz = plt.axes([0.25, 0.25, 0.65, 0.01], facecolor='b')

        axfocus2 = plt.axes([0.25, 0.1, 0.65, 0.01], facecolor='b')
        axtz2 = plt.axes([0.25, 0.15, 0.65, 0.01], facecolor='b')

        focus = Slider(axfocus, 'Focus', 1.0, 3000, valinit=200.0)
        tz = Slider(axtz, 'tz', 0.1, 5.0, valinit=1.0)

        focus2 = Slider(axfocus2, 'Focus2', 1.0, 3000, valinit=self.fx2)
        tz2 = Slider(axtz2, 'tz2', 0.1, 5.0, valinit=self.tz2)

        focus.on_changed(self.update_focus)
        tz.on_changed(self.update_tz)

        focus2.on_changed(self.update_focus2)
        tz2.on_changed(self.update_tz2)

        plt.show()

# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start program")
    mod = PointModel2()
    mod.create_points(15)

    app = Application(mod)
    app.show()

    print("Exit program")

if __name__ == '__main__':
    main()
