import sys
import io
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Workaround so that orb compute doesn't crash
cv2.ocl.setUseOpenCL(False)


def main():
    imagefile = sys.argv[1]
    mapfile = sys.argv[2]
    keyframefile = sys.argv[3]
    keyframe_infos = sys.argv[4]

    image = cv2.imread(imagefile,0)
    # Initiate ORB detector
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    orb = cv2.ORB_create(1000, 1.4, 8, fastThreshold=20)
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kps, image_descriptors = orb.compute(image, kp)

    with io.open(mapfile, 'r') as f:
        pointmap = json.JSONDecoder().decode(f.read())

    with io.open(keyframe_infos, 'r') as f:
        keyframe_infos = json.JSONDecoder().decode(f.read())


    tmp_descriptors = list(map(lambda point: point['descriptors'], pointmap))
    map_descriptors = np.array(tmp_descriptors, dtype=np.uint8)

    # check the descriptor from the json file
#    keyframe_descriptors = np.asarray(keyframe_infos['descriptors'], dtype=np.uint8)
#    keyframe_positions = np.asarray(keyframe_infos['position'])
#    kf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#    kf_matches = kf_matcher.match(keyframe_descriptors, map_descriptors)
#    kf_match_list = list(map(lambda match: match.queryIdx, kf_matches))
#    kf_subset = keyframe_descriptors[kf_match_list]
#    keyframe_positions_subset = keyframe_positions[kf_match_list]
#
#
#    # match the orb points with the points from the map
#    image_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#    image_matches = image_matcher.match(image_descriptors, kf_subset)
#    train_matches = list(map(lambda match: match.trainIdx, image_matches))
#    image_match_list = list(map(lambda match: match.queryIdx, image_matches))

    # match the orb points with the points from the map
    image_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    image_matches = image_matcher.match(image_descriptors, map_descriptors)
    train_matches = list(map(lambda match: match.trainIdx, image_matches))
    image_match_list = list(map(lambda match: match.queryIdx, image_matches))
    image_subset = image_descriptors[image_match_list]
    image_kps_subset = np.asarray(kps)[image_match_list]


    keyframe_descriptors = np.asarray(keyframe_infos['descriptors'], dtype=np.uint8)
    keyframe_positions = np.asarray(keyframe_infos['position'])
    kf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kf_matches = kf_matcher.match(keyframe_descriptors, image_subset)

    kf_kps = np.asarray(list(map(lambda point: cv2.KeyPoint(point[0], point[1], 10),
                                 keyframe_positions)))

    keyframe = cv2.imread(keyframefile, 0)

    sorted_matches = sorted(kf_matches, key=lambda match: match.distance)

#    test_image = image.copy()
#    test_image = cv2.drawKeypoints(keyframe, kf_kps, test_image)
#    plt.imshow(test_image)
#    plt.show()

    out_image = image.copy()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    out_image = cv2.drawMatches(keyframe, kf_kps, image, image_kps_subset, sorted_matches[0:50], out_image)

    plt.imshow(out_image)
    plt.show()


if __name__ == '__main__':
    main()
