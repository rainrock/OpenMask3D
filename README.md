# OpenMask3D
MaskTransformers for Open-Set 3D Scenes

AI Center Group Project, ETH Zurich

References:
1. https://pengsongyou.github.io/openscene
2. https://jonasschult.github.io/Mask3D/
3. https://github.com/openai/CLIP

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

# Visualization [Open3D]

We implement a visualization tool that leverages Open3D to visualize point clouds and process input text queries. This README provides instructions on how to use our visualizer and outlines the functionalities that are yet to be implemented @Jan @Ke


To experience our visualizer, follow these instructions:

1. Navigate to the visualization directory.
    ```
    cd visualization
    ```

2. Run the `visualize.py` script and specify a point cloud file (e.g., `fragment.ply`) to visualize it.
    ```
    python visualize.py fragment.ply
    ```
    Alternatively, you can always choose a file from the menu bar after running the visualizer.
    ```
    python visualize.py
    ```

### Functionalities

There are several functionalities that are yet to be implemented. These functions are assigned to specific team members:

Functions to be Implemented by `@Jan and @Ke`

    compute_clip_feature(text: str) -> clip_feature: This function takes a text input and computes the CLIP feature of the text.

    compute_clip_distance(clip_feature1: clip_feature, clip_feature2: clip_feature) -> distance: This function computes the distance between two different CLIP feature vectors.

    find_similar_points(clip_vector: clip_feature, threshold: float) -> similar_points: This function takes a CLIP vector and returns all the points that have a small enough (under threshold) CLIP distance to the given vector.



Functions to be Implemented by `@Ying`

    render_points_with_color(points: List[point], color: Color) -> None: This function takes the points obtained from the find_similar_points function (Function 3) and renders them with a distinctive color. The rendered points are saved to a new PCD file.

    modify_callback_function() -> None: This function modifies the callback function to reload the scene from the new PCD file.

    update_point_color(points: List[point], color: Color) -> None: This function changes the color of specific points directly and updates only the color information.

