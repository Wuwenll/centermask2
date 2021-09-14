ulimit -s unlimited
ulimit -c unlimited
ulimit -SHn 65536
export MPLCONFIGDIR=".config/matplotlib"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python train_net.py \
    --config-file configs/centermask/SynthText.yaml \
    --num-gpus 6 \
    OUTPUT_DIR training_dir/SynthText36_with_arg \
    MODEL.FCOS.NUM_CLASSES 36 \
    MODEL.RETINANET.NUM_CLASSES 36 \
    MODEL.ROI_HEADS.NUM_CLASSES 36 \
    SOLVER.IMS_PER_BATCH 18 \
    SOLVER.MAX_ITER 300000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    SOLVER.BASE_LR 0.001
