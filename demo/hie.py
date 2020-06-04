# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from fetcher import get_all_imgs_in_dir

# constants
WINDOW_NAME = "COCO detections"

def dump_detection(frame, boxes, scores, classes):
    # frame, -1, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1.
    results = []
    for b, c, s in zip(boxes, classes, scores):
        r = "{},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1".format(frame, b[0], b[1], b[2] - b[0], b[3] - b[1], s)
        results.append(r)
    return results
     
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    videos = ["20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]
    output_dir = "/public/MOTDataset/HIE20/ResNeSt200_hie_all_detections_iou07/test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for v in videos:
        video_sub_dir = "/public/MOTDataset/HIE20/images/test/{}/img".format(v)
        imgs = get_all_imgs_in_dir(video_sub_dir)
        output_file = "{}/{}.txt".format(output_dir, v)
        results = []
        for img in tqdm.tqdm(imgs):
            img_name = img.split("/")[-1]
            frame = img_name.split(".")[0]
            frame = int(frame)
            img = read_image(img, format="BGR")
            predictions, _ = demo.run_on_image(img)
            result_single = dump_detection(frame,
                predictions["instances"].pred_boxes.tensor.tolist(), 
                predictions["instances"].scores.tolist(),
                predictions["instances"].pred_classes.tolist())
            results.extend(result_single)
        with open(output_file, 'w') as f:
            for i in results:
                f.write(i)
                f.write('\n')