import sys
sys.path.append(".")

import numpy as np
import torch
from fusion import feature_fusion
import visualization.clip_utils as clip_utils


def main():

    # load processed instance mask
    processed_mask3d = np.loadtxt("0568/processed_scene0568_00_heatmap.txt") #(200, 232453) np.ndarray min 0, max 1
    print("read instance mask, done")


    # load per-point clip feature
    pc_clip = torch.load('0568/scene0568_00_0.pt')
    pc_features = pc_clip['feat'].numpy()           # (214318, 768) min:-2.1  max 2.3
    mask_ = pc_clip['mask_full'].numpy()             # (232453,)  [True, .. False]
    
    mask = np.asarray([mask_] * 200)                            # (200, 232453)
    processed_mask3d = processed_mask3d[mask] 
    processed_mask3d = np.reshape(processed_mask3d, [200, -1]) # (200, 214318)  
    print("read clip features, done")
    


    # now you have (200, 214318) "processed_mask3d" [0, 1] and (214318, 768) per point CLIP "pc_features" (-2.16, 2.37)

    # per instance clip feature (200, 768)
    feature_fusion(processed_mask3d, pc_features, "0568_00")
    instance_feature = np.loadtxt('test_data/fused_feature_0568_00.txt') # (200, 768) [-1.1, 1.2]
    print("compute instance CLIP features, done")
    
    
    # 
    
    mask = clip_utils.find_mask("table" , processed_mask3d, instance_feature, 'scene0568_00')
    print(mask.shape)

    



if __name__ == "__main__":
    main()
   