import numpy as np
from scipy.special import softmax
import warnings
import os

# Input dimension
# mask3D: Size_Instances * Size_3D_Points   
# clip_feature: Size_3D_Points * Size_Clip_Feature_Vector
# instance_feature: Size_Instances * Size_Clip_Feature_Vector
def feature_fusion(preproessed_mask3d, clip_feature, filename, method = "average"):
    
    size_points = clip_feature.shape[0]
    size_feature = clip_feature.shape[1]
    size_instance = preproessed_mask3d.shape[0]
    
    if len(preproessed_mask3d.shape)>1 and size_points != preproessed_mask3d.shape[1]:
        print(f"clip feature #point:{size_points}, mask3D #points: {preproessed_mask3d.shape[1]}")
        warnings.warn("point cloud sizes don't align.")

    if method == "average":
        instance_feature = average(preproessed_mask3d, clip_feature)
    else:
        instance_feature = None
    
    np.savetxt(f"test_data/fused_feature_{filename}.txt", instance_feature) 
    print(f"Saved fused instance feature to test_data/fused_feature_{filename}.txt")  
    



def average(normalized_mask3d, clip_feature):
    aggregated_clip_feature =  np.matmul(normalized_mask3d, clip_feature)
    sum_of_points = normalized_mask3d.sum(axis = 1).reshape(-1,1)
    return aggregated_clip_feature/sum_of_points