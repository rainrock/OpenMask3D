
import clip
import numpy as np

clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)

def compute_clip_feature(text):

    # generate token
    text = clip.tokenize([text])

    text_features = clip_pretrained.encode_text(text)

    # normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.detach().cpu().numpy()


def compute_clip_distance(feature1, feature2):
    return np.dot(feature1, feature2)/(np.linalg.norm(feature1) * np.linalg.norm(feature2))