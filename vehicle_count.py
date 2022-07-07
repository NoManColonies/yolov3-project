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
import time
from queue import PriorityQueue
from evidence_processor import EvidenceProcessor
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


# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
# print(classNames)
# print(len(classNames))

# Read configurations file
configurations = json.loads(open("config.json").read().strip())
# print(configurations)

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

# work queue
queue = PriorityQueue(10)


def main():
    cameras = []

    for camera in configurations['cameras']:
        name, path, detection_box, traffic_lights = camera.items()
        camera = CameraInstance(name[1], path[1], detection_box[1],
                                traffic_lights[1], configurations['traffic_light_color_boundries'], classNames, colors, queue=queue)
        cameras.append(camera)

    processor = EvidenceProcessor(queue)
    processor.start()

    try:
        while True:
            global success
            for camera_index, camera in enumerate(cameras):
                success = camera.render(net)

            if not success or cv2.waitKey(1) == ord('q'):
                print("success is False. quiting...")
                break
    except BaseException as e:
        print(e)
        sentry_sdk.capture_exception(e)

    for camera in cameras:
        camera.flush()

    queue.join()
    # Finally realese the capture object and destroy all active windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
