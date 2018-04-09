import numpy as np
import cv2
import json

from matplotlib import pyplot as plt

gray = cv2.imread('sample.jpg',0)

# Workaround so that orb compute doesn't crash
cv2.ocl.setUseOpenCL(False)
# Initiate ORB detector
orb = cv2.ORB_create(2000, 1.4, 8, fastThreshold=20)
# find the keypoints with ORB
kp = orb.detect(gray, None)
# compute the descriptors with ORB
kp, des = orb.compute(gray, kp)
# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(gray, kp, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
#plt.imshow(img2), plt.show()

import io
with io.open('test.map', 'r') as f:
    pointmap = json.JSONDecoder().decode(f.read())

descriptors = np.array(list(map(lambda point: point['descriptors'][0], pointmap)), dtype=np.uint8)

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary
#matcher = cv2.FlannBasedMatcher(index_params,search_params)
matcher = cv2.BFMatcher()

matches = matcher.match(des,descriptors)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0]]*len(matches)

import pdb; pdb.set_trace()  # XXX BREAKPOINT
kp_okay = []
# ratio test as per Lowe's paper
#for i in range(0, len(matches)-1):
#    if matches[i].distance < 300:
#        kp_okay.append(kp[i])

for i, match in enumerate(list(matches)):
    if match.distance < 300:
        kp_okay.append(kp[i])

print("okay key points: {}".format(len(kp_okay)))
img2 = cv2.drawKeypoints(gray, kp_okay, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.imshow(img2,),plt.show()
