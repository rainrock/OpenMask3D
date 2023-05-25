import numpy as np
from scipy.special import softmax
import warnings


# Input dimension
# mask3D: Size_Instances * Size_3D_Points   
# clip_feature: Size_3D_Points * Size_Clip_Feature_Vector
def feature_fusion(mask3d, clip_feature, method = "average"):

    size_points = clip_feature.shape[0]
    size_feature = clip_feature.shape[1]
    size_instance = mask3d.shape[0]
    
    if len(mask3d.shape)>1 and size_points != mask3d.shape[1]:
        warnings.warn("point cloud sizes don't align.")
        return None  

    normalized_mask3d = softmax(mask3d - 0.5, axis=0)
    
    if method == "average":
        instance_feature = average(normalized_mask3d, clip_feature)
    else:
        instance_feature = None
    
    return instance_feature



def average(normalized_mask3d, clip_feature):
    return np.matmul(normalized_mask3d, clip_feature)