from evidence_tracker import EvidenceTracker

class ProcessibleEvidence:
    def __init__(self, trackers: list, frames):
        self.trackers = trackers
        self.frames = frames
