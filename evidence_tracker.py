import cv2
import uuid


class EvidenceTracker:
    def __init__(self, index, id):
        self.__positions = []
        self.__starting_frame_index = index
        self.id = id
        self.__positive = False
        self.__evidence_frame = None
        self.uuid = uuid.uuid4()

    def append_position(self, position) -> None:
        self.__positions.append(position)

    def mark_evidence_positive(self, frame) -> None:
        self.__positive = True
        self.__evidence_frame = frame

    def compose_evidence(self, frames) -> None:
        if self.__positive:
            cv2.imwrite(f'result/{self.uuid}.png', self.__evidence_frame)
            print(f'Composing video evidence id: {self.uuid}')
            # out = cv2.VideoWriter(f'result/{self.id}.avi', cv2.VideoWriter_fourcc(
            #     *'XVID'), 10, (1920, 1080))
            out = cv2.VideoWriter(f'result/{self.uuid}.mp4', cv2.VideoWriter_fourcc(
                *'avc1'), 10, (1920, 1080))
            for frame_index, x, y, w, h in self.__positions:
                # print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                cv2.rectangle(frames[frame_index], (x * 2, y * 2),
                              ((x + w) * 2, (y + h) * 2), (0, 0, 255), 2)
                out.write(frames[frame_index])
            out.release()
