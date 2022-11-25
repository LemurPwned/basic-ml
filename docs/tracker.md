# Tracker

Basic tracker implementation suitable for use with decent detectors.
The base library is in C++, so it should not bottleneck (or slow down) significantly the dectection process.

## Usage

### Detection format

The tracker expects detections in the following format:

```
x1, y1, x1, y2, score, ...
```

where `x1`, `y1`, `x2`, `y2` are the bounding box coordinates, `score` is the detection score. The pair `x1, y1` is the top-left corner of the bounding box, and `x2, y2` is the bottom-right corner. You can pass more information after the score, but the tracker will ignore it and return in intact in the output. This information usually contains things such as `class_id` or `frame_id`.

### Tracker initialization

In the tracker initalisation you can pass additional parameters:

```python
tracker = Tracker(
    minShadowCount=3, # minimum number of lacking frames to consider a track as a shadow
    minTrackLength=3, # minimum number of frames to consider a track as a valid track
    iouThreshold=0.3, # IoU threshold for merging tracks
    minConfidenceThreshold=0.3, # minimum confidence threshold for a detection to be considered
)
```

To run the tracker on a sequence of detections, you can use the `update` method:

```python
tracker.init(detections[0])
for frame_i in range(1, n_frames):
    active_tracks = tracker.update(detections[i])
    for track in active_tracks:
        tid = track.getId()
        detection = track.getLastDetection()
        x1, y1, x2, y2, score = detection
```

The `update` method returns a list of active tracks. A track is active if it has been seen in the last `minShadowCount` frames. The `getLastDetection` method returns the last detection associated with the track. The `getId` method returns the track id.

### Track merging

Depending on your detector, you may want to run NMS before passing the detections to the tracker. In this case, you can use the `computeNMS` function:

```python
nms_detections = computeNMS(detections[i], iouThreshold=0.95)
```
