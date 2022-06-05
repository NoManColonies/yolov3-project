import numpy as np
import cv2
import csv
from tracker import *
from sentry_sdk import capture_exception
from traffic_light import TrafficLightColor
from evidence_tracker import EvidenceTracker
from processible_evidence import ProcessibleEvidence
import queue
import time


class CameraInstance:
    def __init__(self, name, path, detection, traffic_lights, boundries, classNames, colors, queue: queue.PriorityQueue):
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
        # Current traffic light color
        self.current_traffic_light = TrafficLightColor.OTHER
        self.previous_traffic_light = TrafficLightColor.OTHER
        self.is_active_camera = False
        self.is_tracking_red_light = False
        self.is_tracking_red_light_waiting_for_reset = False
        self.red_light_tracking_time = time.time()
        # Camera frames for rendering into a video evidences
        self.__evidence_frames = []
        self.__evidence_trackers = []
        self.is_awaiting_flush_command = False
        # camera reconnect attempt
        self.__reconnect_retry_attempt = 0
        # work queue
        self.queue = queue

    def __del__(self):
        self.cap.release()
        self.flush()

    def __connectToCamera(self) -> None:
        # Initialize the videocapture object
        self.cap = cv2.VideoCapture(self.path)

    def __readTrafficLight(self, frame) -> None:
        # create NumPy arrays from the boundaries
        lower = np.array(self.traffic_light_boundries['low'], dtype="uint8")
        upper = np.array(self.traffic_light_boundries['high'], dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        # read traffic light colors from configuration
        red_lights, yellow_lights = self.traffic_lights.items()

        # red light section
        total_red_light_count = len(red_lights[1])
        positive_red_light_count = 0
        required_red_light_count_ratio = 1/2

        for red_traffic_light in red_lights[1]:
            position_x, position_y, width, height = red_traffic_light.items()
            # create blank image for comparison
            blank = np.zeros((width[1], height[1], 3), dtype='uint8')
            traffic_light_mask = output[position_y[1]:
                                        (height[1] + position_y[1]),
                                        position_x[1]:
                                        (width[1] + position_x[1])]

            if not np.array_equal(blank, traffic_light_mask):
                positive_red_light_count = positive_red_light_count + 1
            # cv2.imshow('Red Light', np.hstack([blank, traffic_light_mask]))

        if positive_red_light_count / total_red_light_count >= required_red_light_count_ratio:
            self.previous_traffic_light = self.current_traffic_light
            self.current_traffic_light = TrafficLightColor.RED
            if not self.is_tracking_red_light and not self.is_tracking_red_light_waiting_for_reset:
                print("begin tracking red light...")
                self.is_tracking_red_light = True
                self.red_light_tracking_time = time.time()
            elif round((time.time() - self.red_light_tracking_time), 2) > 5 and not self.is_tracking_red_light_waiting_for_reset:
                print("stop tracking red light. waiting for reset...")
                self.is_tracking_red_light = False
                self.is_tracking_red_light_waiting_for_reset = True
                print(f"deactivating {self.name} camera...")
                self.is_active_camera = False
                self.is_awaiting_flush_command = True
            return
        # yellow light section
        total_yellow_light_count = len(yellow_lights[1])
        positive_yellow_light_count = 0
        required_yellow_light_count_ratio = 1/2

        for yellow_traffic_light in yellow_lights[1]:
            position_x, position_y, width, height = yellow_traffic_light.items()
            # create blank image for comparison
            blank = np.zeros((width[1], height[1], 3), dtype='uint8')
            traffic_light_mask = output[position_y[1]:
                                        (height[1] + position_y[1]),
                                        position_x[1]:
                                        (width[1] + position_x[1])]

            if not np.array_equal(blank, traffic_light_mask):
                positive_yellow_light_count = positive_yellow_light_count + 1
            # cv2.imshow('Yellow Light', np.hstack([blank, traffic_light_mask]))

        if positive_yellow_light_count / total_yellow_light_count >= required_yellow_light_count_ratio:
            self.previous_traffic_light = self.current_traffic_light
            self.current_traffic_light = TrafficLightColor.YELLOW
            return
        # allow for delay between light switching from yellow to red color
        if self.current_traffic_light is TrafficLightColor.YELLOW:
            return
        # abort if red light is currently tracked
        if self.is_tracking_red_light:
            return
        self.previous_traffic_light = self.current_traffic_light
        self.current_traffic_light = TrafficLightColor.OTHER

        if self.is_tracking_red_light_waiting_for_reset:
            # reset red light tracking
            print("resetting red light tracker...")
            self.is_tracking_red_light_waiting_for_reset = False
        # activate camera if not yet
        if not self.is_active_camera:
            print(f"activating {self.name} camera...")
            self.is_active_camera = True

    # Function for finding the center of a rectangle
    def __findCenter(self, x, y, w, h) -> (int, int):
        x1 = int(w/2)
        y1 = int(h/2)
        cx = x+x1
        cy = y+y1
        return cx, cy

    def __processDetectedFrame(self, frame, id) -> None:
        cv2.imwrite(f'result/{self.name}_{id}.png', frame)
        print(f"detected id: {id}")

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
                # self.__processDetectedFrame(copied_original_frame, id)
                for _, tracker in enumerate(self.__evidence_trackers):
                    if tracker.id is id:
                        tracker.mark_evidence_positive(copied_original_frame)
                        break

        # Draw circle in the middle of the rectangle
        cv2.circle(frame, center, 2, (0, 0, 255), -1)

    def __process(self, net, scaled_frame, input_size=320) -> None:
        DETECTION_OFFSET_Y = self.detection['offset_y']
        DETECTION_OFFSET_X = self.detection['offset_x']
        DETECTION_MAX_Y = self.detection['max_y']
        DETECTION_MAX_X = self.detection['max_x']
        MIDDLE_LINE_POSITION = self.detection['line_position_y']
        TOP_LINE_POSITION = self.detection['line_position_y'] - 10
        BOTTOM_LINE_POSITION = self.detection['line_position_y'] + 10

        scaled_original_frame = scaled_frame.copy()
        cropped_detection_frame = scaled_frame[DETECTION_OFFSET_Y:DETECTION_MAX_Y,
                                               DETECTION_OFFSET_X:DETECTION_MAX_X]
        ih, iw, channels = cropped_detection_frame.shape
        blob = cv2.dnn.blobFromImage(
            cropped_detection_frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        # outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        # reason for type checking https://github.com/opencv/opencv/issues/20923
        type_comparison_placeholder = np.zeros(1)
        outputNames = [(layersNames[(i if type(i) is not type(type_comparison_placeholder) else i[0]) - 1])
                       for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
        # Find the objects from the network output
        self.__postProcess(outputs, cropped_detection_frame,
                           scaled_frame, scaled_original_frame)

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
        # Collect evidences
        # self.__evidence_frames.append(original_frame)
        for x, y, w, h, id, _ in boxes_ids:
            for index, tracker in enumerate(self.__evidence_trackers):
                if tracker.id is id:
                    # print('found id')
                    self.__evidence_trackers[index].append_position(
                        (len(self.__evidence_frames) - 1, x, y, w, h))
            else:
                # print('adding new id')
                evidence_tracker = EvidenceTracker(
                    len(self.__evidence_frames) - 1, id)
                evidence_tracker.append_position(
                    (len(self.__evidence_frames) - 1, x, y, w, h))
                self.__evidence_trackers.append(evidence_tracker)
        # Count vehicles
        if self.current_traffic_light is TrafficLightColor.RED:
            for box_id in boxes_ids:
                self.__countVehicle(
                    box_id, frame, original_frame)

    def __resetState(self):
        self.detected_classNames = []
        # Initialize Tracker
        self.tracker = EuclideanDistTracker()
        # List for store vehicle count information
        self.temp_down_list = []
        self.up_list = [0, 0, 0]
        # Camera frames for rendering into a video evidences
        self.__evidence_frames = []
        self.__evidence_trackers = []

    def __draw_debug_output(
        self,
        scaled_frame,
        input_size=320,
        font_color=(0, 0, 255),
        font_size=0.5,
        font_thickness=2,
    ):
        DETECTION_OFFSET_Y = self.detection['offset_y']
        DETECTION_OFFSET_X = self.detection['offset_x']
        DETECTION_MAX_Y = self.detection['max_y']
        DETECTION_MAX_X = self.detection['max_x']
        MIDDLE_LINE_POSITION = self.detection['line_position_y']
        TOP_LINE_POSITION = self.detection['line_position_y'] - 10
        BOTTOM_LINE_POSITION = self.detection['line_position_y'] + 10
        # Draw the crossing lines
        cv2.line(scaled_frame, (DETECTION_OFFSET_X, MIDDLE_LINE_POSITION),
                 (DETECTION_MAX_X, MIDDLE_LINE_POSITION), (255, 0, 255), 2)
        cv2.line(scaled_frame, (DETECTION_OFFSET_X, TOP_LINE_POSITION),
                 (DETECTION_MAX_X, TOP_LINE_POSITION), (0, 0, 255), 2)
        cv2.line(scaled_frame, (DETECTION_OFFSET_X, BOTTOM_LINE_POSITION),
                 (DETECTION_MAX_X, BOTTOM_LINE_POSITION), (0, 0, 255), 2)
        cv2.putText(scaled_frame, "Passed", (110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, font_color, font_thickness)
        cv2.putText(scaled_frame, f"Car:        {str(self.up_list[0])}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(scaled_frame, f"Bus:        {str(self.up_list[1])}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(scaled_frame, f"Truck:      {str(self.up_list[2])}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow(f'{self.name} output', scaled_frame)

    def flush(self):
        print("flushing camera state")
        self.collectStats()
        # Encode evidence frame into a video file
        print(
            f"total evidence trackers in this round: {len(self.__evidence_trackers)}")
        print(
            f"total evidence frames in this round: {len(self.__evidence_frames)}")
        evidences = ProcessibleEvidence(
            self.__evidence_trackers, self.__evidence_frames)
        self.queue.put_nowait(evidences)
        # for tracker in self.__evidence_trackers:
        #     tracker.compose_evidence(self.__evidence_frames.copy())
        # reset camera detection states
        self.__resetState()

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

    def render(
        self,
        net,
    ) -> bool:

        success, frame = self.cap.read()
        # check if video ran out of frame
        if not success:
            if self.__reconnect_retry_attempt <= 9:
                self.cap.release()
                print("camera disconnected. retrying...")
                self.__connectToCamera()
                self.__reconnect_retry_attempt = self.__reconnect_retry_attempt + 3
                return True
            else:
                return False
        elif self.__reconnect_retry_attempt > 0:
            self.__reconnect_retry_attempt = self.__reconnect_retry_attempt - 1
        # resize frame to reduce unnecessary load on gpu
        scaled_frame = cv2.resize(frame.copy(), (0, 0), None, 0.5, 0.5)
        # read current traffic light
        self.__readTrafficLight(scaled_frame)
        # if last color is RED and current is OTHER then flush the camera state
        if self.current_traffic_light is not self.previous_traffic_light:
            print(
                f"current: {self.current_traffic_light}\nprevious: {self.previous_traffic_light}")
        if self.is_awaiting_flush_command:
            print("entering next traffic phrase")
            self.flush()
            self.is_awaiting_flush_command = False
            return success

        if self.is_active_camera:
            if self.current_traffic_light is TrafficLightColor.RED:
                self.__evidence_frames.append(frame)
                self.__process(net, scaled_frame)
            elif self.current_traffic_light is TrafficLightColor.YELLOW:
                self.__evidence_frames.append(frame)
                self.__process(net, scaled_frame)
        
        self.__draw_debug_output(scaled_frame)

        return success
