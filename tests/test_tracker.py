import os
from pathlib import Path
from typing import Union

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from tqdm import tqdm

# from basic_ml.vis import annotate_frame
from basic_ml.tracker import ByteTracker, IOUTracker, computeIOU

ROOT = Path(os.path.dirname(__file__))/Path("/Volumes/KINGSTON/data/MOT16/train/")

"""
Convention here is:
    - x, y, w, h are the bounding box coordinates
    - conf is the confidence score
"""

def annotate_frame(frame, tracks):
    # print(len(tracks))
    for track in tracks:
        box = track.getLastDetection()
        x1, y1, x2, y2, _ = box
        # print("\t", box)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2.putText(frame, str(track.getId()), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0), 2, cv2.LINE_AA)    

    return frame

def read_mot_txt(filename, min_score=0.0):
    """Read MOT txt file."""
    data = pd.read_csv(filename, sep=',', header=None,
    names=[
         "frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"
    ])
    data.sort_values(by=["frame", "id"], inplace=True)
    data = data.loc[data['conf'] > min_score]
    return data

def get_box(data, frame):
    box_data = data.loc[data["frame"] == frame]
    bbox= box_data[["x", "y", "w", "h", "conf"]].values 
    id_val = box_data["id"].values
    bbox = np.asarray([
        [x[0], x[1], x[0] + x[2], x[1] + x[3], x[4]] for x in bbox
    ])
    return bbox, id_val

class TestDetector:
    def __init__(self, detection_root) -> None:
        self.detection_root = detection_root 
        self.detections = read_mot_txt(self.detection_root)

    def detect(self, frame_i):
        box, _ = get_box(self.detections, frame_i)
        return box

def score_tracker(gt_hypotheses, tracker_hypotheses):
    """
    Score tracker hypotheses against ground truth hypotheses
    """
    # Create cost matrix
    cost_matrix = np.zeros((len(gt_hypotheses), len(tracker_hypotheses)))
    for i, gt_hypothesis in enumerate(gt_hypotheses):
        for j, tracker_hypothesis in enumerate(tracker_hypotheses):
            cost_matrix[i, j] = 1. - computeIOU(gt_hypothesis, tracker_hypothesis)

    return cost_matrix
    

def test_iou_tracker(maxShadowCount=100, minTrackLength=1, iouThreshold=0.646, silent=False):
    """
    Test IOU tracker performance on MOT16 train set.
    """
    return tracker_test_runner(
        lambda: IOUTracker(maxShadowCount=maxShadowCount, minTrackLength=minTrackLength, iouThreshold=iouThreshold),
        silent=silent
    )

def test_byte_tracker(silent=False):
    """
    Test Byte tracker performance on MOT16 train set.
    """
    return tracker_test_runner(
        lambda: ByteTracker(),
        silent=silent
    )


def tracker_test_runner(tracker_init: Union[IOUTracker, ByteTracker], 
                        silent=False, det_folder=None, detector_impl=TestDetector, tracker_params={},
                        vis_folder=None):
    """
    Test tracker performance on MOT16 train set.
    """
    names, acc_list = [], []
    # take a smaller subset of the data for testing
    # mot_range = (9, 10, 11, 13)
    mot_range = (9, )
    for mot_num in mot_range:
        filename = ROOT / f"MOT16-{mot_num:02d}"
        if vis_folder:
            os.makedirs(os.path.join(vis_folder, f"MOT16-{mot_num:02d}"), exist_ok=True)
        if det_folder is None:
            det_folder = "gt/gt.txt"
        detector = detector_impl(filename / det_folder)
        names.append(filename.name)
        acc = mm.MOTAccumulator(auto_id=True)
        gt = read_mot_txt(filename/'gt'/'gt.txt')
        frames = gt["frame"].max() + 1
        tracker = tracker_init(**tracker_params)
        for frame in tqdm(range(1, frames), desc=f"Processing {filename.name}", disable=silent):
            gt_detections, gt_ids = get_box(gt, frame)
            gt_detections = gt_detections.astype(np.float32).tolist()
            detections = detector.detect(frame).astype(np.float32).tolist()
            tracks = tracker.update(detections)
            tracker_dets = [track.getLastDetection() for track in tracks]
            cost_matrix = score_tracker(gt_detections, tracker_hypotheses=tracker_dets)
            tracker_ids = [track.getId() for track in tracks]
            acc.update(gt_ids, tracker_ids, cost_matrix.tolist())
            if vis_folder:
                FRAME_FOLDER = filename / "img1" / f"{frame:06d}.jpg"
                img  = cv2.imread(str(FRAME_FOLDER))
                img = annotate_frame(img, tracks)
                cv2.imwrite(os.path.join(vis_folder, f"MOT16-{mot_num:02d}", f"{frame:06d}.jpg"), img)
        acc_list.append(acc)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        acc_list, 
        names=names
    )
    if not silent:
        print(summary)
    return summary

def optimise_tracker_performance():
    """
    Run optimisation to find best tracker parameters.
    """
    from bayes_opt import BayesianOptimization

    opt_bounds = {
        'maxShadowCount': (0, 100),
        'minTrackLength': (0, 10),
        'iouThreshold': (0.4, 0.99),
    }

    def score_tracker(maxShadowCount, minTrackLength, iouThreshold):
        summary = test_iou_tracker(maxShadowCount=int(maxShadowCount), 
        minTrackLength=int(minTrackLength), 
        iouThreshold=iouThreshold, silent=True)
        return summary['motp'].mean()

    optimiser = BayesianOptimization(
        f=score_tracker,
        pbounds=opt_bounds,
        random_state=1,
    )
    optimiser.maximize(
        init_points=10,
        n_iter=30,
    )

if __name__ == "__main__":
    # res = tracker_test_runner(tracker_init=IOUTracker, 
    #                         #   det_folder="det/det.txt", 
    #                           vis_folder='./scratch/IOUTracker',
    #                           tracker_params={
    #                                 'maxShadowCount': 0,
    #                           },
    #                           detector_impl=TestDetector)
    res = tracker_test_runner(tracker_init=ByteTracker, 
                            #   det_folder="det/det.txt", 
                              vis_folder='./scratch/ByteTracker',
                              tracker_params={
                                    # 'minConfidenceThreshold': 0.8,
                              },
                              detector_impl=TestDetector)
    # optimise_tracker_performance()
