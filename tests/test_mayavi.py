# Author: Prabhu Ramachandran <prabhu at aero dot iitb dot ac dot in>
# Copyright (c) 2007, Enthought, Inc.
# License: BSD style.
import numpy as np
from mayavi import mlab

x, y, z = np.random.randn(3, 100)
mlab.points3d(x, y, z)
mlab.show()
