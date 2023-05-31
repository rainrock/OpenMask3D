import sys
sys.path.append(".")
import torch
import numpy as np
import open3d as o3d


def main():

    mask_ = torch.load('0568/scene0568_00_0.pt')['mask_full'].numpy()             # (232453,)  [True, .. False]
    
    data = np.load("0568/0568_00.npy")      #  (232453, 12)
    data = data[mask_]                      #  (214318, 3)
    positions = data[:, :3] 
    colors = data[:, 3:6] / 255.0  # Normalize colors to [0, 1] range (assuming they are in the [0, 255] range)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Normalize RGB values to the range [0, 1]
    o3d.io.write_point_cloud('0568/0568_pc.ply', point_cloud)
    
if __name__ == "__main__":
    main()
   