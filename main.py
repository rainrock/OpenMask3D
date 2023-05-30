import argparse
import subprocess
import os
import numpy as np
from fusion import feature_fusion

def main(args):
    # Your code logic goes here
    print(f"Input file: {args.input}")
    
    if args.verbose:
        print("Verbose mode enabled")

    # Call Adams thingy here
    # TODO @Adam
    processed_mask3d = subprocess.Popen([["python3", "mask3d.py"]])

    # # Do preprocessing
    # p1 = subprocess.Popen(["python3", "preprocess/preprocess_3d_replica.py"])
    # p2 = subprocess.Popen(["python3", "preprocess/preprocess_2d_replica.py"])

    # exit_codes = [p.wait() for p in [p1, p2]]

    # # Get Clip features
    # p3 = subprocess.Popen(["python3", "clip_features/replica_openseg.py"])

    # exit_codes = [p.wait() for p in [p0, p3]]

    # Compute clip feature per instance prediction    
    
    # read precomputed clip feature
    point_cloud_clip_feature = np.load(args.clip_feature_path)
    
    # compute clip feature per instance
    clip_feature_per_instance = feature_fusion(processed_mask3d, point_cloud_clip_feature)
    
    query = "" # get the query somehow
    # Compute the feature for the query

    #p4 = subprocess.Popen(["python3", "clip_features/clip_text_encoder.py", "--input", "{}".format(query)])

    # output

    # TODO @Ying

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Your program description")

    # Add arguments to the parser
    parser.add_argument("clip_feature_path", help="Input file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function
    main(args)

