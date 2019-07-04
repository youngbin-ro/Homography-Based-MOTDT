import cv2
import numpy as np

def Homography(img, prev_img):

    if prev_img is None:
        return

    orb = cv2.ORB_create()
    kpt1, des1 = orb.detectAndCompute(prev_img, None)
    kpt2, des2 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M
