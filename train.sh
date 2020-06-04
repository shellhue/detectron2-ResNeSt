

#python tools/train_net.py --num-gpus 8 --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=".:${PYTHONPATH}"
export DETECTRON2_DATASETS="/home/huangzeyu/detectron2/datasets"

# python tools/train_net.py \
#     --num-gpus=1 \
#     --config-file=configs/COCO-Detection/faster_rcnn_ResNeSt_50_FPN_1x.yaml \
#     SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x-e9955232.pth
    
# python tools/train_net.py \
#     --num-gpus=1 \
#     --config-file=configs/COCO-Detection/faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x.yaml \
#     --eval-only \
#     MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_50_FPN_syncbn_range-scale-1x-e9955232.pth

# python tools/train_net.py \
#     --num-gpus=8 \
#     --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     --eval-only \
#     MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x-1be2a87e.pth

# python projects/hie_tmp/train_net.py \
#     --num-gpus=8 \
#     --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     --eval-only \
#     MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x-1be2a87e.pth

# python tools/train_net.py \
#     --num-gpus=2 \
#     --dist-url='tcp://127.0.0.1:50160' \
#     --config-file=configs/COCO-Detection/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x-1be2a87e.pth

python projects/hie_tmp/train_net.py \
    --num-gpus=6 \
    --dist-url='tcp://127.0.0.1:50160' \
    --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
    SOLVER.IMS_PER_BATCH 6 MODEL.WEIGHTS faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x-1be2a87e.pth OUTPUT_DIR output/resnest_hie_part