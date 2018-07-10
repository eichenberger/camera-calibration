import io
import json

class SlpMap:
    def __init__(self, map_file):
        self.map_file = map_file
        self.map_json = None
        self.points2d = None
        self.points3d = None
        self.descriptors = None

    def load_map(self):
        self.map_json
        map_json = None
        with io.open(self.map_file, 'r') as f:
            self.map_json = json.JSONDecoder().decode(f.read())

    def get_points_kf(self, kf_id):
        points2d = []
        points3d = []
        descriptors = []
        kf_index = -1
        kf = None
        for i,temp_kf in enumerate(self.map_json['KeyFrames']):
            if temp_kf['id'] == kf_id:
                kf_index = i
                kf = temp_kf

        if kf is None:
            print('Keyframe index not found')
            return

        for mp in self.map_json['MapPoints']:
            if kf_id in mp['observingKfIds']:
                observing_kf_idx = list.index(mp['observingKfIds'], kf_id)
                kp_idx = mp['corrKpIndices'][observing_kf_idx]
                points3d.append(mp['mWorldPos']['data'])
                points2d.append(kf['keyPtsUndist'][kp_idx])
                descriptor = kf['featureDescriptors']['data'][kp_idx*32:(kp_idx+1)*32]
                descriptors.append(descriptor)

        self.points2d = points2d
        self.points3d = points3d
        self.descriptors = descriptors

