import numpy as np
import cv2

image = cv2.imread('color_sample.png')

boundaries = [
    # ([17, 15, 100], [50, 56, 200]),
    ([0, 19, 124], [214, 204, 255]),
]

for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
	# find the colors within the specified boundaries and apply
	# the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    blank = np.zeros((4, 4, 3), dtype='uint8')
    traffic_light1 = output[33:37,274:278]
    cv2.imshow('Output', traffic_light1)
    cv2.imshow('Blank', blank)
    traffic_light2 = output[11:15,381:385]
    cv2.imshow('Output2', traffic_light2)
    traffic_light3 = output[11:15,392:396]
    cv2.imshow('Output3', traffic_light3)
    print(np.array_equal(blank, traffic_light1))
    print(np.array_equal(blank, traffic_light2))
    print(np.array_equal(blank, traffic_light3))
	# show the images
    # cv2.imshow('Mask', output)
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)