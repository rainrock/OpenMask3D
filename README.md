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
   
   Given Clip feature space $V$ and instance class space $`X`$ , $|X| = 200$(?)
   
   Input: 
   
   Given 3D Scan point cloud set $P \subset R^3$, for each point $p \in P$ we have
        
   - [OpenScene] 3D clip feature, a mapping $\phi$ ($P \to V$) for each point to Clip feature space $V$.
   <!---     
   - Mask3D heatmap, a mapping for each point to probability space over instance class $X$ , $\sum_{x \in X}Pr(p \in x) = 1$
   -->
   - [Mask3D] Heatmaps $h_i$ from $P \to [0.5, 1]^{|P|}$, where $0 \leq i < |X|$, $h_i$ is a mapping from each point $p$ to the confidence level of this point belonging to instance class $x \in X$.
   
   Normalization:
   - $g_i = softmax(h_i - 0.5) : P \to [0, 1]^{|P|}$, where $i$ stands for the $i$-th instance class in $X$.
   
   Output:
   
   - for each instance class $x \in X$, compute the aggregated clip feature over all points: $\sum_{p \in P} \phi (p) g_i(p)$
  

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
