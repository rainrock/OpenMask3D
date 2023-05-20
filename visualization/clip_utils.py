import numpy as np
import matplotlib.pyplot as plt

# by Jan & Ke
# compute clip feature vector from a text input
def compute_textCLIP(text_input):
    return np.ones(768)


# by Jan & Ke
def find_mask(text_input, file_name):
    #  find features corresponding to the file_name
    name = file_name.split("/")[-1].replace('.ply', "")
    print("this function will read the the CLIP features of pcd from: ", name)
    features = None

    text_CLIP = compute_textCLIP(text_input)

    # for all features, find the distance between text_CLIP

    # scale them between 0 to 1

    
    return np.zeros(2000)


# by Ying
def generate_color_map(mask):
    # Normalize the mask values to the range [0, 1]
    normalized_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    
    # Create a color map using the normalized mask values
    colormap = plt.cm.get_cmap('YlGnBu')  # Yellow to Green to Blue color map
    
    # Map normalized mask values to RGB colors
    colors = colormap(normalized_mask)
    
    return colors[:, :3]  # Extract only the RGB values, excluding the alpha channel



    