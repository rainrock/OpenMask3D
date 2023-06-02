# OpenMask3D
MaskTransformers for Open-Set 3D Scenes

AI Center Group Project, ETH Zurich

References:
1. https://pengsongyou.github.io/openscene
2. https://jonasschult.github.io/Mask3D/
3. https://github.com/openai/CLIP

Public Dataset:
1. Openscene Dataset: https://cvg-data.inf.ethz.ch/openscene/data/
2. Replica:


# Installation 
For Preprocessing, Feature fusion and Clip text-encoding you only need to get
    torch
    tensorflow
    numpy
    imageio
    tqdm
    clip

Then, you can place your data in data/Replica/.

Moreover, you need an OpenSeg Model. Just download it from [here](https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view?usp=sharing)
and put it inside the folder openseg.

# Feature fusion  (OpenScene and Mask3D)

Definition
   
   Given Clip feature space $V$ and instance space $`X`$ , $|X| = 200$(?)
   
   Input: 
   
   Given 3D Scan point cloud set $P \subset R^3$, for each point $p \in P$ we have
        
   - [OpenScene] 3D clip feature, a mapping $\phi$ ($P \to V$) for each point to Clip feature space $V$.
   <!---     
   - Mask3D heatmap, a mapping for each point to probability space over instance class $X$ , $\sum_{x \in X}Pr(p \in x) = 1$
   -->
   - [Mask3D] Heatmaps $h_i$ from $P \to [0.5, 1]^{|P|}$, where $1 \leq i \leq |X|$, $h_i$ is a mapping from each point $p$ to the confidence level of this point belonging to instance $x \in X$.
   
   Normalization:
   - $g_i = \frac{h_i - 0.5}{\sum_{i} h_i - 0.5} : P \to [0, 1]^{|P|}$, where $i$ stands for the $i$-th instance in $X$.
   
   Output:
   
   - for the $i$-th instance in $X$, compute the normalized aggregated clip feature over all points: $\frac{\sum_{p \in P} \phi (p) g_i(p)}{\sum_{p \in P} {g_i(p)}}$

-----------------------------------------------------------------------------------------
With precomputed mask3D heatmaps and Openscene clip feature, running the following command will produce the visualization mask for 3D point cloud for certain scene:
    
    python3 main.py -clip_feature_path scene_clip_feature.pt -mask3d_path scene/heatmap/ -scene_name scenename
    
e.g. for scene 0568_00

    python3 main.py -clip_feature_path scene0568_00_0.pt -mask3d_path 0568_00_mask_heatmap/heatmap/ -scene_name scene0568_00


# Visualization [Open3D]

We implement a visualization tool that leverages Open3D to visualize point clouds and process input text queries. This README provides instructions on how to use our visualizer and outlines the functionalities that are yet to be implemented @Jan @Ke


To experience our visualizer, run the following script. 

Choose a `.ply` file from the menu bar after running the visualizer.

    python visualization/visualize.py
    

Alternatively, specify the path to a point cloud file (e.g., fragment.ply) as an argment.
 
    python visualization/visualize.py visualization/fragment.ply
    

### Functionalities

There are several functionalities that are yet to be implemented. These functions are assigned to specific team members:
@Ke` 

Implement the function "`find_mask`" in `visualization/clip_utils.py`.

`Input`:  "text"   (e.g, "sofa")

`OutPut`: PointCloud Mask ($|P|$ values from 0 to 1 indicate how relevant the points are to the given text). (e.g.[0.1, 0.3, 0, 0,...1] of size $|P|$)

    find_similar_points(clip_vector: clip_feature, threshold: float) -> similarity mask: This function takes a CLIP vector and returns a mask value from 0 to 1. 
    large value indicates the point CLIP is close to the text CLIP. 
