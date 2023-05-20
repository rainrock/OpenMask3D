import numpy as np


# compute clip feature vector from a text input
def compute_textCLIP(text_input):
    return np.ones(768)



def find_mask(text_input, file_name):
    #  find features corresponding to the file_name
    name = file_name.split("/")[-1].replace('.ply', "")
    print("this function will read the the CLIP features of pcd from: ", name)
    features = None

    text_CLIP = compute_textCLIP(text_input)

    # for all features, find the distance between text_CLIP

    # scale them between 0 to 1

    
    return np.zeros(2000)




    