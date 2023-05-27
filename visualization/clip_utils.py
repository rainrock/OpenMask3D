import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from clip_features.clip_text_encoder import compute_clip_distance, compute_clip_feature

# by Jan & Ke
# compute clip feature vector from a text input
def compute_textCLIP(text_input):
    return compute_clip_feature(text_input)


# Input
# clip_feature: Size_3D_Points * Size_Clip_Feature_Vector
# mask3D: Size_Instances * Size_3D_Points   
# Output
# mask: Size_Instances * Size_3D_Points
def find_mask(text_input, file_instance_feature, file_mask3D):
    #  find features corresponding to the file_name
    #name = file_name.split("/")[-1].replace('.ply', "")
    #print("this function will read the the CLIP features of pcd from: ", name)
    instance_feature = np.loadtxt(file_instance_feature)
    mask3D = np.loadtxt(file_mask3D)
    normalized_mask3D = softmax(mask3D - 0.5, axis=0)

    text_CLIP = compute_textCLIP(text_input)

    # for all features, find the distance between text_CLIP    
    normalized_dist = np.asarray([compute_clip_distance(feature, text_CLIP) for feature in instance_feature])
    
    mask = np.asarray([mask3D[i]* normalized_dist[i] for i in range(len(normalized_dist))])
    
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



    