# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *
import json
from camera import *
import sentry_sdk
from os import getenv
from dotenv import load_dotenv
# load .env file
load_dotenv()
# config sentry sdk
sentry_sdk.init(
    getenv("SENTRY_URL", ""),

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=float(getenv("SENTRY_TRACE_RATE", "1.0"))
)

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('video_day1_hires.mp4')
# cap = cv2.VideoCapture('rtsp://admin:Hh5943610@10.54.2.2:554/Streaming/Channels/101')
input_size = 320

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
# middle_line_position = 225
middle_line_position = 105
# up_line_position = middle_line_position - 15
# down_line_position = middle_line_position + 15
up_line_position = middle_line_position - 10
down_line_position = middle_line_position + 10


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
# print(classNames)
# print(len(classNames))

# Read configurations file
configurations = json.loads(open("config.json").read().strip())
# print(configurations)

# class index for our required detection classes
# required_class_index = [2, 3, 5, 7]
required_class_index = [2, 5, 7]

detected_classNames = []

# Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0]
down_list = [0, 0, 0]

# Function for count vehicle


def count_vehicle(box_id, img, original_img):
    copied_original_img = original_img.copy()
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    # if (iy > up_line_position) and (iy < middle_line_position):
    #     if id not in temp_up_list:
    #         temp_up_list.append(id)

    if iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1
            cv2.rectangle(copied_original_img, (x, y),
                          (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(f'result/{id}.png', copied_original_img)

    # elif iy > down_line_position:
    #     if id in temp_up_list:
    #         temp_up_list.remove(id)
    #         down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output


DETECTION_OFFSET_Y = 60
DETECTION_OFFSET_X = 190
DETECTION_MAX_Y = 358
DETECTION_MAX_X = 425


def postProcess(outputs, cropped_img, img, original_img):
    global detected_classNames
    height, width = cropped_img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    boxes.append([x + DETECTION_OFFSET_X, y +
                                 DETECTION_OFFSET_Y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidence_scores, confThreshold, nmsThreshold)
    # check if video ran out of frame
    if not hasattr(indices, 'flatten'):
        return
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img, original_img)


def realTime():
    while True:
        success, img = cap.read()
        # check if video ran out of frame
        if not success:
            break
        # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        copied_original_img = img.copy()
        cropped_img = img[DETECTION_OFFSET_Y:DETECTION_MAX_Y,
                          DETECTION_OFFSET_X:DETECTION_MAX_X]
        ih, iw, channels = cropped_img.shape
        blob = cv2.dnn.blobFromImage(
            cropped_img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        # outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputNames = [(layersNames[i - 1])
                       for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)

        # Find the objects from the network output
        postProcess(outputs, cropped_img, img, copied_original_img)

        # Draw the crossing lines

        # cv2.line(img, (0, middle_line_position),
        #          (iw, middle_line_position), (255, 0, 255), 2)
        # cv2.line(img, (0, up_line_position),
        #          (iw, up_line_position), (0, 0, 255), 2)
        # cv2.line(img, (0, down_line_position),
        #          (iw, down_line_position), (0, 0, 255), 2)
        cv2.line(img, (DETECTION_OFFSET_X, middle_line_position),
                 (DETECTION_MAX_X, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (DETECTION_OFFSET_X, up_line_position),
                 (DETECTION_MAX_X, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (DETECTION_OFFSET_X, down_line_position),
                 (DETECTION_MAX_X, down_line_position), (0, 0, 255), 2)

        # # Draw counting texts in the frame
        # cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #             font_size, font_color, font_thickness)
        # cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #             font_size, font_color, font_thickness)
        # cv2.putText(img, "Car:        "+str(up_list[0])+"     " + str(
        #     down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     " + str(
        #     down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Bus:        "+str(up_list[2])+"     " + str(
        #     down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Truck:      "+str(up_list[3])+"     " + str(
        #     down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Passed", (110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[1]), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(up_list[2]), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'bus', 'truck'])
        up_list.insert(0, "Up")
        # down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        # cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    cameras = []
    active_camera_index = -1

    for camera in configurations['cameras']:
        name, path, detection_box, traffic_lights = camera.items()
        cameras.append(CameraInstance(name[1], path[1], detection_box[1],
                       traffic_lights[1], configurations['traffic_light_color_boundries'], classNames, colors))

    while True:
        global success
        for camera_index, camera in enumerate(cameras):
            success, traffic_light_status = camera.render(
                net,
                is_active_camera=camera_index is active_camera_index
            )
            # set this camera to active camera if traffic light turn green
            if traffic_light_status is TrafficLightColor.OTHER:
                # flush the state of last camera if there is a last camera
                if active_camera_index != -1:
                    cameras[active_camera_index].flush()
                # set current active camera to this camera
                active_camera_index = camera_index

        if not success or cv2.waitKey(1) == ord('q'):
            break
    # Finally realese the capture object and destroy all active windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # realTime()
    main()
