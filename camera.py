import numpy as np
import cv2
import csv
from tracker import *


class CameraInstance:
    def __init__(self, name, path, detection, traffic_lights, boundries, classNames, colors):
        self.name = name
        self.path = path
        self.detection = detection
        self.traffic_lights = traffic_lights
        self.traffic_light_boundries = boundries
        self.detected_classNames = []
        self.classNames = classNames
        self.colors = colors
        # Initialize the videocapture object
        self.__connectToCamera()
        # Initialize Tracker
        self.tracker = EuclideanDistTracker()
        # List for store vehicle count information
        self.temp_down_list = []
        self.up_list = [0, 0, 0]

    def __del__(self):
        self.collectStats()
        self.cap.release()

    def __connectToCamera(self) -> None:
        # Initialize the videocapture object
        self.cap = cv2.VideoCapture(self.path)

    def __readTrafficLight(self, frame) -> bool:
        # create NumPy arrays from the boundaries
        lower = np.array(self.traffic_light_boundries['low'], dtype="uint8")
        upper = np.array(self.traffic_light_boundries['high'], dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        total_count = len(self.traffic_lights)
        positive_count = 0
        required_count_ratio = 2/3

        for traffic_light in self.traffic_lights:
            position_x, position_y, width, height = traffic_light.items()
            # create blank image for comparison
            blank = np.zeros((width[1], height[1], 3), dtype='uint8')
            traffic_light_mask = output[position_y[1]:
                                        (height[1] + position_y[1]),
                                        position_x[1]:
                                        (width[1] + position_x[1])]

            if not np.array_equal(blank, traffic_light_mask):
                positive_count = positive_count + 1

        return positive_count / total_count >= required_count_ratio

    # Function for finding the center of a rectangle
    def __findCenter(self, x, y, w, h) -> (int, int):
        x1 = int(w/2)
        y1 = int(h/2)
        cx = x+x1
        cy = y+y1
        return cx, cy

    def __processDetectedFrame(self, frame, id) -> None:
        cv2.imwrite(f'result/{self.name}_{id}.png', frame)

    def __countVehicle(self, box_id, frame, original_frame) -> None:
        MIDDLE_LINE_POSITION = self.detection['line_position_y']
        TOP_LINE_POSITION = self.detection['line_position_y'] - 10
        BOTTOM_LINE_POSITION = self.detection['line_position_y'] + 10
        copied_original_frame = original_frame.copy()
        x, y, w, h, id, index = box_id

        # Find the center of the rectangle for detection
        center = self.__findCenter(x, y, w, h)
        ix, iy = center

        if iy < BOTTOM_LINE_POSITION and iy > MIDDLE_LINE_POSITION:
            if id not in self.temp_down_list:
                self.temp_down_list.append(id)

        elif iy < TOP_LINE_POSITION:
            if id in self.temp_down_list:
                self.temp_down_list.remove(id)
                self.up_list[index] = self.up_list[index]+1
                cv2.rectangle(copied_original_frame, (x, y),
                              (x + w, y + h), (0, 0, 255), 2)
                self.__processDetectedFrame(copied_original_frame, id)

        # Draw circle in the middle of the rectangle
        cv2.circle(frame, center, 2, (0, 0, 255), -1)

    def __postProcess(self, outputs, cropped_frame, frame, original_frame,
                      confThreshold=0.2,
                      nmsThreshold=0.2,
                      required_class_index=[2, 5, 7]
                      ) -> None:
        # global detected_classNames
        DETECTION_OFFSET_Y = self.detection['offset_y']
        DETECTION_OFFSET_X = self.detection['offset_x']
        height, width = cropped_frame.shape[:2]
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
                        x, y = int((det[0]*width)-w /
                                   2), int((det[1]*height)-h/2)
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

            color = [int(c) for c in self.colors[classIds[i]]]
            name = self.classNames[classIds[i]]
            self.detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(frame, f'{name.upper()} {int(confidence_scores[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            detection.append(
                [x, y, w, h, required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = self.tracker.update(detection)

        for box_id in boxes_ids:
            self.__countVehicle(
                box_id, frame, original_frame)

    def collectStats(self) -> None:
        # Write the vehicle counting information in a file and save it
        with open(f"{self.name}_data.csv", 'w') as f1:
            cwriter = csv.writer(f1)
            cwriter.writerow(['Direction', 'car', 'bus', 'truck'])
            self.up_list.insert(0, "Passed")
            cwriter.writerow(self.up_list)
        f1.close()
        self.up_list = [0, 0, 0]
        self.temp_down_list = []

    def render(self, net, input_size=320, font_color=(0, 0, 255), font_size=0.5, font_thickness=2
               ) -> (bool, bool):
        DETECTION_OFFSET_Y = self.detection['offset_y']
        DETECTION_OFFSET_X = self.detection['offset_x']
        DETECTION_MAX_Y = self.detection['max_y']
        DETECTION_MAX_X = self.detection['max_x']
        MIDDLE_LINE_POSITION = self.detection['line_position_y']
        TOP_LINE_POSITION = self.detection['line_position_y'] - 10
        BOTTOM_LINE_POSITION = self.detection['line_position_y'] + 10

        success, frame = self.cap.read()
        # check if video ran out of frame
        if not success:
            return success, False

        traffic_light_status = self.__readTrafficLight(frame)

        if traffic_light_status:
            # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            copied_original_frame = frame.copy()
            cropped_detection_frame = frame[DETECTION_OFFSET_Y:DETECTION_MAX_Y,
                                            DETECTION_OFFSET_X:DETECTION_MAX_X]
            ih, iw, channels = cropped_detection_frame.shape
            blob = cv2.dnn.blobFromImage(
                cropped_detection_frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            net.setInput(blob)
            layersNames = net.getLayerNames()
            # outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
            outputNames = [(layersNames[i - 1])
                           for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            outputs = net.forward(outputNames)

            # Find the objects from the network output
            self.__postProcess(outputs, cropped_detection_frame,
                               frame, copied_original_frame)

        # Draw the crossing lines
        cv2.line(frame, (DETECTION_OFFSET_X, MIDDLE_LINE_POSITION),
                 (DETECTION_MAX_X, MIDDLE_LINE_POSITION), (255, 0, 255), 2)
        cv2.line(frame, (DETECTION_OFFSET_X, TOP_LINE_POSITION),
                 (DETECTION_MAX_X, TOP_LINE_POSITION), (0, 0, 255), 2)
        cv2.line(frame, (DETECTION_OFFSET_X, BOTTOM_LINE_POSITION),
                 (DETECTION_MAX_X, BOTTOM_LINE_POSITION), (0, 0, 255), 2)
        cv2.putText(frame, "Passed", (110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, font_color, font_thickness)
        cv2.putText(frame, f"Car:        {str(self.up_list[0])}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(frame, f"Bus:        {str(self.up_list[1])}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(frame, f"Truck:      {str(self.up_list[2])}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow(f'{self.name} output', frame)

        return success, traffic_light_status
