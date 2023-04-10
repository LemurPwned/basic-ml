import cv2


def annotate_frame(frame, tracks):
    for track in tracks:
        box = track.getLastDetection()
        x1, y1, x2, y2, _ = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, str(track.getId()), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return frame
