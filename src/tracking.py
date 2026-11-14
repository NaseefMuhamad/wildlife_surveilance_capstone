"""Associate detections across consecutive frames of a single camera."""


def overlap(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0


class DetectionTracker:
    """Match nearby boxes of the same class, with one ID per visible object."""

    def __init__(self, min_overlap=0.30, max_missing=3):
        self.min_overlap = min_overlap
        self.max_missing = max_missing
        self.frame = 0
        self.next_id = 1
        self.active = {}

    def update(self, detections):
        self.frame += 1
        self.active = {
            track_id: (box, last_seen)
            for track_id, (box, last_seen) in self.active.items()
            if self.frame - last_seen <= self.max_missing + 1
        }
        candidates = []
        for index, detection in enumerate(detections):
            for track_id, (box, _) in self.active.items():
                if detection["class"] == box["class"]:
                    score = overlap(detection["xyxy"], box["xyxy"])
                    if score >= self.min_overlap:
                        candidates.append((-score, track_id, index))

        assigned = {}
        used_ids = set()
        for _, track_id, index in sorted(candidates):
            if index not in assigned and track_id not in used_ids:
                assigned[index] = track_id
                used_ids.add(track_id)

        tracked = []
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            box = {**detection, "track_id": track_id}
            self.active[track_id] = (box, self.frame)
            tracked.append(box)
        return tracked
