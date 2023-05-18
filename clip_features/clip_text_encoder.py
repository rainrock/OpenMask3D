import os
import numpy as np
import clip
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='DisNet')
    parser.add_argument('--input', type=str)
    parser.add_argument('--out_dir', type=str, default='./', help='specify the output directory')
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print('Loading the CLIP model...')
    clip_pretrained, _ = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    print('Finished loading.')
    print('Ready for queries')

    query_string = args.input

    # generate token
    text = clip.tokenize([query_string])

    text_features = clip_pretrained.encode_text(text)

    # normalize
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # save features
    np.save(os.path.join(out_dir, '{}.npy'.format(query_string)), text_features.detach().cpu().numpy())


if __name__ == '__main__':
    main()
