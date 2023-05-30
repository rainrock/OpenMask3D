import argparse
import numpy as np
import torch
from fusion import feature_fusion
import mask3d
from visualization.clip_utils import find_mask

def main(args):
    # Your code logic goes here
    print(f"Input file: {args.clip_feature_path}")
    print(f"Input dir: {args.mask3d_path}")
    print(f"Scene: {args.scene_name}")

    
    if args.verbose:
        print("Verbose mode enabled")


    # Call Adams thingy here
    # TODO @Adam
    #processed_mask3d = subprocess.Popen([["python3", "mask3d.py"]])
    processed_mask3d = mask3d.read(args.mask3d_path + args.scene_name)

    # # Do preprocessing
    # p1 = subprocess.Popen(["python3", "preprocess/preprocess_3d_replica.py"])
    # p2 = subprocess.Popen(["python3", "preprocess/preprocess_2d_replica.py"])

    # exit_codes = [p.wait() for p in [p1, p2]]

    # # Get Clip features
    # p3 = subprocess.Popen(["python3", "clip_features/replica_openseg.py"])

    # exit_codes = [p.wait() for p in [p0, p3]]

    # Compute clip feature per instance prediction    
    
    # read precomputed clip feature
    point_cloud_clip_feature = torch.load(args.clip_feature_path)['feat'].numpy()

    # read mask for clip_features
    mask = torch.load(args.clip_feature_path)['mask_full'].numpy()
    mask = np.asarray([mask] * 200)

    processed_mask3d = processed_mask3d[mask]
    processed_mask3d = np.reshape(processed_mask3d, [200, -1])

    # compute clip feature per instance
    clip_feature_per_instance = feature_fusion(processed_mask3d, point_cloud_clip_feature, args.scene_name)
    
    query = "table" # get the query somehow
    # Compute the feature for the query
    
    # compute the mask for visualization
    mask = find_mask(query, args.scene_name)

    #p4 = subprocess.Popen(["python3", "clip_features/clip_text_encoder.py", "--input", "{}".format(query)])

    # output

    # TODO @Ying

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments to the parser
    parser.add_argument("-clip_feature_path", help="Input file path")
    parser.add_argument("-mask3d_path", help="Input Dir path")
    parser.add_argument("-scene_name", help="scene name")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    # Parse the command-line arguments
    args = parser.parse_args()
    
    

    # Call the main function
    main(args)

