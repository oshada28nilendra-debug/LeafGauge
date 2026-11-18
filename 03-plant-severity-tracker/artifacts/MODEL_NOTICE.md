# Model provenance

`synthetic_model.npz` was fitted to procedural images only and accompanies the MIT-licensed code.

`real_apple_model.npz` was fitted to the declared ATLDSD training and validation subset. This artifact is distributed for non-commercial research under CC BY-NC-SA 4.0. Retain dataset attribution to Jingze Feng and Xiaofei Chao, Apple Tree Leaf Disease Segmentation Dataset, Science Data Bank, 2022, https://doi.org/10.11922/sciencedb.01627. Source dataset license: https://creativecommons.org/licenses/by-nc-sa/4.0/.

The artifact contains a learned linear pixel head, feature scaling, thresholds and provenance metadata. ImageNet-pretrained ResNet18 weights are downloaded separately through torchvision; they are not included in this archive. Neither model establishes field reliability. See VALIDATION.md for actual measured limitations.
