import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import softmax
from clip_features.clip_text_encoder import compute_clip_distance, compute_clip_feature


# by Jan & Ke
# compute clip feature vector from a text input
def compute_textCLIP(text_input):
    return compute_clip_feature(text_input)

# processed_mask3d  (200, 214318)  
# instance_feature  (200, 768)

# return: a mask of size (214318, ) shows how close each point to text clip
def find_mask(text_input, processed_mask3d, instance_feature, filename):
    text_CLIP = compute_textCLIP(text_input)[0]
    normalized_dist = np.asarray([compute_clip_distance(feature, text_CLIP) for feature in instance_feature])

    mask = np.asarray([processed_mask3d[i]* normalized_dist[i] for i in range(len(normalized_dist))])

    Threshold = mask.max() * 0.8
    
    mask = mask[mask.max(axis = 1)> Threshold]
    mask = np.max(mask, axis=0)
    return mask


# Input
# clip_feature: Size_3D_Points * Size_Clip_Feature_Vector
# mask3D: Size_Instances * Size_3D_Points   
# Output
# mask: Size_Instances * Size_3D_Points
# filename: the name of a precomputed scene
def find_mask_(text_input, processed_mask3d, filename):
    #name = file_name.split("/")[-1].replace('.ply', "")
    #print("this function will read the the CLIP features of pcd from: ", name)
    
    
    print(f"Reading fused instance feature from test_data/fused_feature_{filename}.txt")  
    instance_feature = np.loadtxt(f"test_data/fused_feature_{filename}.txt")
    
    #print(f"Reading preprocessed heatmap from test_data/processed_{filename}_heatmap.txt")  
    #mask3D = np.loadtxt(f"test_data/processed_{filename}_heatmap.txt")
    
    text_CLIP = compute_textCLIP(text_input)[0]

    # for all features, find the distance between text_CLIP    
    normalized_dist = np.asarray([compute_clip_distance(feature, text_CLIP) for feature in instance_feature])
    
    # map the clip distance back to the heatmap mask, 
    # mapping method could be changed
    mask = np.asarray([processed_mask3d[i]* normalized_dist[i] for i in range(len(normalized_dist))])
    
    # Thresholding, remove mask with little relevance
    Threshold = mask.max() * 0.8
    
    mask = mask[mask.max(axis = 1)> Threshold]
    
    print("vis mask: ", mask.shape)
    
    print("Successfully compute the visualization mask for 3D point cloud!")
    
    np.savetxt(f"test_data/vis_mask_{filename}.txt", mask)
    
    print(f"Saved 3D point cloud visualization mask to test_data/vis_mask_{filename}.txt")
    
    return mask


# by Ying
def generate_color_map(mask):
    # Normalize the mask values to the range [0, 1]
    normalized_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    
    # Create a color map using the normalized mask values
    colormap = plt.cm.get_cmap('YlGnBu')  # Yellow to Green to Blue color map
    
    # Map normalized mask values to RGB colors
    colors = colormap(normalized_mask)
    
    return colors[:, :3]  # Extract only the RGB values, excluding the alpha channel



    