import torch

import numpy as np

from voxelizer import Voxelizer

from MinkowskiEngine import SparseTensor

SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                      np.pi))
TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

ROTATION_AXIS = 'z'
LOCFEAT_IDX = 2

voxel_size=0.05

model_path = ""

# get model paint
model = None

checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
try:
    model.load_state_dict(checkpoint['state_dict'], strict=True)

fn = ''
out_path = ''

locs_in, feats_in, labels_in = torch.load(fn)
labels_in[labels_in == -100] = 255
labels_in = labels_in.astype(np.uint8)
if np.isscalar(feats_in) and feats_in == 0:
    # no color in the input point cloud, e.g nuscenes lidar
    feats_in = np.zeros_like(locs_in)
else:
    feats_in = (feats_in + 1.) * 127.5

vox = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=TRANSLATION_AUGMENTATION_RATIO_BOUND)

locs, feats, labels, inds_reconstruct, vox_ind = vox.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)

coords = torch.from_numpy(locs).int()
feats = torch.from_numpy(feats).float() / 127.5 - 1.
inds_reverse = torch.from_numpy(inds_reconstruct).long()


sinput = SparseTensor(feats, coords)

predictions = model(sinput)
predictions = predictions[inds_reverse, :]
pred = predictions.half() @ text_features.t()
logits_pred = torch.max(pred, 1)[1].cpu()