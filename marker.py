import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 500)

# Add white border
bordered = np.ones((600, 600), dtype=np.uint8) * 255
bordered[50:550, 50:550] = marker_img

cv2.imwrite("hardhat_aruco_sticker.png", bordered)
print("✅ Saved! Print at exactly 10cm × 10cm")