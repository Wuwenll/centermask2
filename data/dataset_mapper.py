# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import pdb

from detectron2 import data
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.structures import BoxMode
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from .detection_utils import build_augmentation
from .augmentation import CustomedAugInput, RandomRotation, RandomCropWithInstance, ColorJitter, RandomBlur
import pickle

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger("detectron2")


class DatasetMapperWithBasis(DatasetMapper):
    
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train) 
        if is_train:
            # angle, expand=True, center=None, sample_style="range", interp=None
            if cfg.INPUT.RANDOM_CROP.ENABLED:
                self.augmentation.insert(
                    0,
                    RandomCropWithInstance(
                        crop_type=cfg.INPUT.RANDOM_CROP.CROP_TYPE,
                        crop_size=cfg.INPUT.RANDOM_CROP.CROP_SIZE,
                        crop_instance=cfg.INPUT.RANDOM_CROP.CROP_INSTANCE,
                    )
                )
                logger.info(
                    "Crop used in training: " + str(self.augmentation[0])
                )
            if cfg.INPUT.ROTATE.ENABLED:
                self.augmentation.insert(
                    0,
                    RandomRotation(
                        angle=cfg.INPUT.ROTATE.ANGLE,
                        sample_style=cfg.INPUT.ROTATE.SAMPLE_STYLE
                    ),
                )
                logger.info(
                    "Rotation used in training: " + str(self.augmentation[0])
                )
            if cfg.INPUT.RANDOM_BLUR.ENABLED:
                self.augmentation.insert(
                    0,
                    RandomBlur(
                        kernel_size=cfg.INPUT.RANDOM_BLUR.KERNEL_SIZE,
                        possibility=cfg.INPUT.RANDOM_BLUR.POSSIBILITY,
                    ),
                )
                logger.info(
                    "Blur used in training: " + str(self.augmentation[0])
                )
            if cfg.INPUT.COLOR_JITTER.ENABLED:
                self.augmentation.insert(
                    0,
                    ColorJitter(
                        brightness=cfg.INPUT.COLOR_JITTER.BRIGHTNESS,
                        contrast=cfg.INPUT.COLOR_JITTER.CONTRAST,
                        saturation=cfg.INPUT.COLOR_JITTER.SATURATION,
                        hue=cfg.INPUT.COLOR_JITTER.HUE
                    ),
                )
                logger.info(
                    "Color Jitter used in training: " + str(self.augmentation[0])
                )
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        
        aug_input = CustomedAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))  

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # print(dataset_dict['file_name'])
            # filename = dataset_dict['file_name'].split('/')[-1]
            # import cv2
            # bboxes = [b['bbox'].tolist() for b in annos]
            # aug_image = image.copy()
            # for b in bboxes:
            #     cv2.rectangle(aug_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            # polys = [seg['segmentation'][0].reshape(-1, 2).astype(np.int32) for seg in annos]
            # cv2.polylines(aug_image, polys, True, (0, 255, 0), 2)
            # cv2.imwrite(filename, aug_image)
            
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
