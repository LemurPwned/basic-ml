import numpy as np
import pandas as pd


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
