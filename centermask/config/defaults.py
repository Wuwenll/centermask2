# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.DATALOADER.SAMPLE_RATIOS = None

_C.MODEL.MOBILENET = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.VOVNET = CN()

_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# CenterMask
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD.ASSIGN_CRITERION = "area"
_C.MODEL.MASKIOU_ON = False
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0

_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.NAME = "MaskIoUHead"
_C.MODEL.ROI_MASKIOU_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASKIOU_HEAD.NUM_CONV = 4


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.ROI_KEYPOINT_HEAD.ASSIGN_CRITERION = "ratio"

_C.INPUT.ROTATE = CN()
_C.INPUT.ROTATE.ENABLED = True
_C.INPUT.ROTATE.ANGLE = [-15, 15]
_C.INPUT.ROTATE.SAMPLE_STYLE = "range"

_C.INPUT.RANDOM_CROP = CN()
_C.INPUT.RANDOM_CROP.ENABLED = True
_C.INPUT.RANDOM_CROP.CROP_TYPE = "relative_range"
_C.INPUT.RANDOM_CROP.CROP_SIZE = 0.8
_C.INPUT.RANDOM_CROP.CROP_INSTANCE = False

_C.INPUT.COLOR_JITTER = CN()
_C.INPUT.COLOR_JITTER.ENABLED = True
_C.INPUT.COLOR_JITTER.BRIGHTNESS = 0.
_C.INPUT.COLOR_JITTER.CONTRAST = 0.
_C.INPUT.COLOR_JITTER.SATURATION = 0.
_C.INPUT.COLOR_JITTER.HUE = 0.

_C.INPUT.RANDOM_BLUR = CN()
_C.INPUT.RANDOM_BLUR.ENABLED = True
_C.INPUT.RANDOM_BLUR.KERNEL_SIZE = 3
_C.INPUT.RANDOM_BLUR.POSSIBILITY = 0.2
