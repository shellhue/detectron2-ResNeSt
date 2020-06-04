export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=".:${PYTHONPATH}"
export DETECTRON2_DATASETS="/home/huangzeyu/detectron2/datasets"

python demo/hie.py \
	--config-file=projects/hie_tmp/faster_cascade_rcnn_ResNeSt_200_FPN_syncbn_range-scale_1x.yaml \
	--opts MODEL.WEIGHTS output/resnest_hie_all/model_0049999.pth MODEL.ROI_HEADS.NMS_THRESH_TEST 0.7