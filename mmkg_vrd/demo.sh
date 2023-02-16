#!/bin/bash -e
CUDA_VISIBLE_DEVICES=5 \
    python tools/demo/demo_image.py \
    --config_file sgg_configs/vrd/R152FPN_vrd_reldn.yaml \
    --img_file custom_io/imgs/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg \
    --save_file custom_io/out/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg \
    --visualize_relation \
    MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False \
    MODEL.WEIGHT custom_io/ckpt/RX152FPN_reldn_oi_best.pth \
    2>&1 | tee logs/demo.log > /dev/null &




# srun -p ai4science \
#     --job-name=demo \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=4 \