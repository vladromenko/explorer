"""Small shared transform metric for hand-eye acceptance tools."""
import numpy as np
from scipy.spatial.transform import Rotation

def transform_error(predicted,observed):
    error=np.linalg.inv(predicted)@observed
    return float(np.linalg.norm(error[:3,3])),float(np.degrees(Rotation.from_matrix(error[:3,:3]).magnitude()))
