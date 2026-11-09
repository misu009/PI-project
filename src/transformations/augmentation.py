import config
# Numpy
import numpy as np
# Pytorch
import torch
from torchvision import transforms
# import Albumentations package
import albumentations as A
# Import pytorch utilities from albumentations
from albumentations.pytorch import ToTensorV2
# Modificarea importului pentru Normalize conform versiunilor noi
from albumentations import Normalize


def get_list_of_transformations(probability):

    if config.APPROACH == 'pretrained':
        
        # Pre-trained
        return [
            A.HorizontalFlip(p = probability), # apply horizontal flip to 50% of images
            A.OneOf(
                [
                    # apply one of transforms to % of the images
                    A.RandomBrightnessContrast(), # apply random contrast
                    A.RandomGamma(), # apply random gamma
                    A.RandomBrightnessContrast(), # apply random brightness # Only 3 channels
                ],
                p = probability
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2), # Only 3 channels: TypeError: clahe supports only uint8 inputs 
                    # IAASharpen and IAAEmboss are deprecated/removed, replaced with Sharpen and Emboss
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(), # Only 3 channels
                ], 
                p = probability
            ),
            # A.OneOf(
            #     [
            #         A.ElasticTransform(),
            #         A.GridDistortion(),
            #     ], 
            #     p = probability
            # ),
            A.OneOf(
                [
                    A.CenterCrop(height=150, width=150),
                    A.RandomCrop(height=150, width=150),
                ], 
                p = probability
            ),
            # Convert the image to PyTorch tensor
            # ToTensorV2(),
            # Normalize
            # Normalize(mean=config.TRAIN_IMG_MEAN[plane], std=config.TRAIN_IMG_STD[plane]),
        ]

    elif config.APPROACH == 'slices':
        
        return [
            A.HorizontalFlip(p = probability), # apply horizontal flip to 50% of images
            A.OneOf(
                [
                    A.RandomBrightnessContrast(), # apply random contrast
                    A.RandomGamma(), # apply random gamma
                ],
                p = probability
            ),
            A.OneOf(
                [
                    A.Sharpen(),
                    A.Emboss(),
                ], 
                p = probability
            ),
            A.OneOf(
                [
                    A.CenterCrop(height=150, width=150),
                    A.RandomCrop(height=150, width=150),
                ], 
                p = probability
            ),
        ]

def get_albumentations_group_pipeline(additional_targets, train, probability=0.5, plane=''):
    """ Define the augmentation pipeline """

    list_of_transformations = get_list_of_transformations(probability)

    if train == 'train' and probability > 0:

        return A.Compose(
            
            list_of_transformations,

            p = 1,

            additional_targets=additional_targets
        )

    else:

        return A.Compose(
            [
                # Normalize
                # Normalize(mean=config.TRAIN_IMG_MEAN[plane], std=config.TRAIN_IMG_STD[plane]),
            ],

            p = 1,

            additional_targets=additional_targets
        )


def augment_images(array, train='train', how='', probability=0.5, plane=''):
    """ Data Augmentation """

    if how == 'albumentations-group': # same transformation to all images on the same MRI

        # Validation and test sets do not get normalization anymore
        if train != 'train' or probability == 0.0: return array

        # 3. Albumentations: same transformation to all slices
        # print('[AUGMENTATION] Albumentations transformation will be applied (same for all slices)')

        # Same transformation to all the images
        targets = {}
        for i in range(len(array[1:])):
            targets['image' + str(i + 1)] = 'image'

        # The transformation pipeline
        # Training: full pipeline
        # Validation: only normalization
        custom_augment = get_albumentations_group_pipeline(targets, train, probability, plane)

        # Dictionary of images
        d = {}
        d['image'] = array[0]
        for i in range(len(array[1:])):
            d['image' + str(i + 1)] = array[i + 1]

        # Custom transformation
        out = custom_augment(**d)

        # Grab the images
        out_array = []
        out_array.append(out['image'])
        for i in range(len(array[1:])):
            out_array.append(out['image' + str(i + 1)])

        # Stack them
        array = np.array(out_array)

    elif how != '':

        raise ValueError('Augmentation technique {} is unknown'.format(how))

    # else:

    #     print('[AUGMENTATION] No augmentation is being done!')

    return array