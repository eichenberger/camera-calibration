class CameraParams:
    def __init__(self, cameramodel=None, camermodel_est=None):
        if cameramodel != None:
            self.fx = cameramodel['f'][0]
            self.fy = cameramodel['f'][1]
            self.cx = cameramodel['c'][0]
            self.cy = cameramodel['c'][1]
            self.thetax = cameramodel['theta'][0]
            self.thetay = cameramodel['theta'][1]
            self.thetaz = cameramodel['theta'][2]
            self.tx = cameramodel['t'][0]
            self.ty = cameramodel['t'][1]
            self.tz = cameramodel['t'][2]
            self.k1 = cameramodel['k'][0]
            self.k2 = cameramodel['k'][1]
            self.k3 = cameramodel['k'][2]
            self.p1 = cameramodel['p'][0]
            self.p2 = cameramodel['p'][1]

    @staticmethod
    def _add_param(space,input, param):
        output = input + param
        output = output + " "*(space - len(param))
        return output

    def get_string(self, space, digits=3):
        output = ""
        output = self._add_param(space, output, str(round(self.fx,digits)))
        output = self._add_param(space, output, str(round(self.fy,digits)))
        output = self._add_param(space, output, str(round(self.cx,digits)))
        output = self._add_param(space, output, str(round(self.cy,digits)))
        output = self._add_param(space, output, str(round(self.thetax,digits)))
        output = self._add_param(space, output, str(round(self.thetay,digits)))
        output = self._add_param(space, output, str(round(self.thetaz,digits)))
        output = self._add_param(space, output, str(round(self.tx,digits)))
        output = self._add_param(space, output, str(round(self.ty,digits)))
        output = self._add_param(space, output, str(round(self.tz,digits)))
        output = self._add_param(space, output, str(round(self.k1,digits)))
        output = self._add_param(space, output, str(round(self.k2,digits)))
        output = self._add_param(space, output, str(round(self.k3,digits)))
        output = self._add_param(space, output, str(round(self.p1,digits)))
        output = self._add_param(space, output, str(round(self.p2,digits)))
        return output

    def get_as_array(self):
        return [self.fx, self.fy, self.cx, self.cy,
                self.thetax, self.thetay, self.thetaz,
                self.tx, self.ty, self.tz,
                self.k1, self.k2, self.k3, self.p1, self.p2]
