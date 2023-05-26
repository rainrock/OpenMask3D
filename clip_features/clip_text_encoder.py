import os
import numpy as np
import clip

def compute_clip_feature(text):
    clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    # generate token
    text = clip.tokenize([text])

    text_features = clip_pretrained.encode_text(text)

    # normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_features.detach().cpu().numpy()
