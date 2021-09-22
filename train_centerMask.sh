ulimit -s unlimited
ulimit -c unlimited
ulimit -SHn 65536
export MPLCONFIGDIR=".config/matplotlib"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python train_net.py \
    --config-file configs/centermask/SynthText.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/62-zoom-in-color \
    MODEL.FCOS.NUM_CLASSES 62 \
    MODEL.RETINANET.NUM_CLASSES 62 \
    MODEL.ROI_HEADS.NUM_CLASSES 62 \
    SOLVER.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 300000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    SOLVER.BASE_LR 0.001
