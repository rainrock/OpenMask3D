import argparse
import numpy as np
import os
from scipy.special import softmax


# read precomputed Mask3d heatmaps

# precomputed heatmaps are named as {scene name}_{0} to {scene name}_{201}
# e.g. for heatmaps ../dir/scene0707_00_{0:201}, filepath should be ../dir/scene0707_00

LAST_FILE_NR = 201
FILEPATH = ""

# Output dimension: Size_Instances * Size_3D_Points
def read_mask3d(FILEPATH):
    print(f"Start reading {FILEPATH}_{0} to {FILEPATH}_{LAST_FILE_NR}")
    
    index = 0
    fname = f"{FILEPATH}_{index}.txt"
    
    mask3d = np.loadtxt(fname)
    index += 1

    while index <= LAST_FILE_NR:
        fname = f"{FILEPATH}_heatmaps/{FILEPATH}_{index}.txt" 
        next_instance = np.loadtxt(f"{FILEPATH}_{index}.txt")
        mask3d = np.vstack((mask3d, next_instance))
        index += 1
        fname = f"{FILEPATH}_{index}.txt"

        
    print("Successfully read precomputed heatmaps!")
        
    return mask3d

def preprocess_heatmap_softmax(mask3d):
    
    normalized_mask3d = softmax(mask3d - 0.5, axis=0)
    
    return normalized_mask3d


def preprocess_heatmap_average(mask3d):
    mask3d -= 0.5
    normalized_mask3d = mask3d / mask3d.sum(axis = 0)
    
    return normalized_mask3d
    
    

def main():
    # # Parser
    # parser = argparse.ArgumentParser(description="Read precomputed heatmaps")
    # parser.add_argument("filepath", help="Input file path")
    
    # # Parse the command-line arguments
    # args = parser.parse_args()
    
    unprocessed_heatmap = read_mask3d(FILEPATH=FILEPATH)
    preprocessed_heatmap = preprocess_heatmap_average(unprocessed_heatmap)

    print(unprocessed_heatmap.shape)
        
    filename = os.path.basename(FILEPATH)
    np.savetxt(f"test_data/processed_{filename}_heatmap.txt", preprocessed_heatmap)
    
    print(f"Save preprocessed heatmap to test_data/processed_{filename}_heatmap.txt")
    
    return preprocessed_heatmap


if __name__ == "__main__":
    main()