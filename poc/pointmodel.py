import multiprocessing as mp
import numpy as np
import json
import io

class PointModel:
    """docstring for PointModel"""
    def __init__(self, model='model.json'):
        with io.open(model, "r") as fd:
            self.model = json.load(fd)


    def _draw_line(self, a, b, n):
        points = np.zeros((n,3))
        a = np.array(a)
        b = np.array(b)
        points[0,:] = a
        points[-1,:] = b
        dist = (b - a)/(n - 1)
        p = a
        for i in range(1,n-1):
            p = p + dist
            points[i,:] = p

        return points

    def create_points(self, n):
        model = self.model
        self.points = np.ones((len(self.model["connections"])*n, 4))
        i = 0
        for connection in model["connections"]:
            c0 = model["points"][connection[0]]["value"]
            c1 = model["points"][connection[1]]["value"]
            self.points[i:i+n,0:3] = self._draw_line(c0, c1, n)
            i = i + n

    def _show_model(self):
        from mayavi import mlab
        points = self.points
        mlab.points3d(points[:,0], points[:,1], points[:,2], mode='axes',
                      color=(0,1,0), scale_mode='none', scale_factor=0.1)
        mlab.show()

    def show_model(self):
        plot_proc = mp.Process(target=self._show_model)
        plot_proc.start()
        self.plot_proc = plot_proc

