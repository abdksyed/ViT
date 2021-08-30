"""
---
title: Vision Transformer (ViT)
summary: >
 A PyTorch implementation/tutorial of the paper
 "An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale"
---

#  Vision Transformer (ViT)
This is a [PyTorch](https://pytorch.org) implementation of the paper
[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://papers.labml.ai/paper/2010.11929).

Vision transformer applies a pure transformer to images without any convolution layers.
They split the image into patches and apply a transformer on patch embeddings.
[Patch embeddings](#PathEmbeddings) are generated by applying a simple linear transformation
to the flattened pixel values of the patch.
Then a standard transformer encoder is fed with the patch embeddings, along with a
classification token `[CLS]`.
The encoding on the `[CLS]` token is used to classify the image with an MLP.
When feeding the transformer with the patches, learned positional embeddings are
added to the patch embeddings, because the patch embeddings do not have any information
about where that patch is from.
The positional embeddings are a set of vectors for each patch location that get trained
with gradient descent along with other parameters.
ViTs perform well when they are pre-trained on large datasets.
The paper suggests pre-training them with an MLP classification head and
then using a single linear layer when fine-tuning.
The paper beats SOTA with a ViT pre-trained on a 300 million image dataset.
They also use higher resolution images during inference while keeping the
patch size the same.
The positional embeddings for new patch locations are calculated by interpolating
learning positional embeddings.

This work is inspired and also adopted from the great work of
[LabML ViT Implementaion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/vit/__init__.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list

import collections.abc

def to_2tuple(x):
    """
    Converts the single value 'x' to a tuple of ('x', 'x')
    """
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

# Variables for Image Size and Patch Size
img_size = 224
patch_size = 16

# img_size -> (224,224), path_size -> (16,16)
img_size = to_2tuple(img_size)
patch_size = to_2tuple(patch_size)

# Batch of Images.
# Each image is a tensor of shape (3, 224, 224)
x = torch.rand((32, 3, 224, 224))

# Converitng Image to Patch Embeddings.
# We create a convolution layer with a kernel size and and stride length equal to patch size.
# This is equivalent to splitting the image into patches and doing a linear
# transformation on each patch. \n
# y -> (32, 768, 14, 14)
# After Flattening & Transpose y -> (32, 196, 768) 

proj = nn.Conv2d(3, 768, 16, 16)
y = proj(x)
y = y.flatten(2).transpose(1, 2)


class PatchEmbeddings(nn.Module):
    """
    <a id="PatchEmbeddings">
    ## Get patch embeddings
    </a>

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    
    This class implementation is the combined of the above steps.
    Arguments:
        image_size: The size of the image to split into patches.
        patch_size: The size of the patches to split the image into.
        num_channels: The number of channels in the image.
        embed_dim: The dimension of the embeddings.
    Returns:
        A `PatchEmbeddings` module.
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x



