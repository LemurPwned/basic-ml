import os
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from tqdm import tqdm

from basic_ml.tracker import Tracker

ROOT = Path(os.path.dirname(__file__))/Path("../data/MOT16/train/")

"""
Convention here is:
    - x, y, w, h are the bounding box coordinates
    - conf is the confidence score
"""

def read_mot_txt(filename):
    """Read MOT txt file."""
    data = pd.read_csv(filename, sep=',', header=None,
    names=[
         "frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"
    ])
    data = data.loc[data['conf'] == 1]
    return data

def get_box(data, frame):
    box_data = data.loc[data["frame"] == frame]
    return box_data[["x", "y", "w", "h", "conf"]].values, box_data["id"].values

class TestDetector:
    def __init__(self, detection_root) -> None:
        self.detection_root = detection_root 
        self.detections = read_mot_txt(self.detection_root)

    def detect(self, frame_i):
        return get_box(self.detections, frame_i)

def iou(bbox1, bbox2):
    """
    Compute the intersection over union of two bounding boxes.
    """
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    # Compute union
    union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection
    # Compute IoU
    iou = intersection / union
    return iou

def score_tracker(gt_hypotheses, tracker_hypotheses):
    """
    Score tracker hypotheses against ground truth hypotheses
    """
    # Create cost matrix
    cost_matrix = np.zeros((len(gt_hypotheses), len(tracker_hypotheses)))
    for i, gt_hypothesis in enumerate(gt_hypotheses):
        for j, tracker_hypothesis in enumerate(tracker_hypotheses):
            cost_matrix[i, j] = 1. - iou(gt_hypothesis, tracker_hypothesis)

    return cost_matrix
    

def test_tracker(maxShadowCount=0, minTrackLength=3, iouThreshold=0.5, silent=False):
    """
    Test tracker performance on MOT16 train set.
    """
    names, acc_list = [], []
    # take a smaller subset of the data for testing
    mot_range = (10, 11, 12, 13)
    for mot_num in mot_range:
        filename = ROOT / f"MOT16-{mot_num:02d}"
        names.append(filename.name)
        acc = mm.MOTAccumulator(auto_id=True)
        gt = read_mot_txt(filename/'gt'/'gt.txt')
        frames = gt["frame"].max() + 1
        tracker = Tracker(maxShadowCount=maxShadowCount, minTrackLength=minTrackLength, iouThreshold=iouThreshold)
        for frame in tqdm(range(1, frames), desc=f"Processing {filename.name}", disable=silent):
            gt_detections, gt_ids = get_box(gt, frame)
            tracker.update(gt_detections.astype(np.float32).tolist())
            tracks = tracker.getActiveTracks()
            tracker_dets = [track.getLastDetection() for track in tracks]
            cost_matrix = score_tracker(gt_detections, tracker_hypotheses=tracker_dets)
            trakcer_ids = [track.getId() for track in tracks]
            acc.update(gt_ids, trakcer_ids, cost_matrix.tolist())
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
    opt_bounds = {
        'maxShadowCount': (0, 100),
        'minTrackLength': (0, 10),
        'iouThreshold': (0.4, 0.99),
    }

    def score_tracker(maxShadowCount, minTrackLength, iouThreshold):
        summary = test_tracker(maxShadowCount=int(maxShadowCount), 
        minTrackLength=int(minTrackLength), 
        iouThreshold=iouThreshold, silent=False)
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
    # test_tracker()
    optimise_tracker_performance()
