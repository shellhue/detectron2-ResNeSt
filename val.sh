# python projects/hie_tmp/train_net.py \
#     --num-gpus=6 \
#     --dist-url='tcp://127.0.0.1:50160' \
#     --resume \
#     --eval-only \
#     --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     SOLVER.IMS_PER_BATCH 6 OUTPUT_DIR output/resnest_hie_all



# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python projects/hie_tmp/train_net.py \
#     --num-gpus=8 \
#     --dist-url='tcp://127.0.0.1:50161' \
#     --eval-only \
#     --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     MODEL.WEIGHTS output/resnest_hie_all/model_0049999.pth MODEL.ROI_HEADS.NMS_THRESH_TEST 0.7

# hie all            nms0.5     nms0.7
# model_0064999.pth  24.6725    25.717
# model_0059999.pth  24.6699
# model_0049999.pth  24.7422
# model_0039999.pth  24.6738
# model_0029999.pth  25.5605
# model_0019999.pth  25.6140
# model_0009999.pth  26.6076    27.835
# model_0004999.pth  25.4944


# python projects/hie_tmp/train_net.py \
#     --num-gpus=8 \
#     --dist-url='tcp://127.0.0.1:50161' \
#     --eval-only \
#     --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
#     MODEL.WEIGHTS output/resnest_hie_part/model_0029999.pth MODEL.ROI_HEADS.NMS_THRESH_TEST 0.7

# hie_part_val       nms0.5     nms0.7
# model_0009999.pth  -          30.663
# model_0029999.pth  -          26.7919


# hie_part_test      nms0.5     nms0.7
# model_0009999.pth  -          28.2192
# model_0029999.pth  -          25.718


python projects/hie_tmp/train_net.py \
    --num-gpus=8 \
    --eval-only \
    --resume \
    --dist-url='tcp://127.0.0.1:50160' \
    --config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
    SOLVER.IMS_PER_BATCH 8 OUTPUT_DIR output/resnest_hie_pretrain DATASETS.TRAIN '("pretrain_and_hie_train",)' DATASETS.TEST '("pretrain_and_hie_test",)' MODEL.ROI_HEADS.NMS_THRESH_TEST 0.7

# pretrain_and_hie_test     nms0.5     nms0.7
# model_0099999.pth         21.247     22.0123