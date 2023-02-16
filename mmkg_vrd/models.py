# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import os.path as op
import argparse
import json
from pathlib import Path
import os

from mmkg_vrd.scene_graph_benchmark.scene_parser import SceneParser
from mmkg_vrd.scene_graph_benchmark.AttrRCNN import AttrRCNN
from mmkg_vrd.maskrcnn_benchmark.data.transforms import build_transforms
from mmkg_vrd.maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from mmkg_vrd.maskrcnn_benchmark.config import cfg
from mmkg_vrd.scene_graph_benchmark.config import sg_cfg
from mmkg_vrd.maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from mmkg_vrd.maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from mmkg_vrd.maskrcnn_benchmark.utils.miscellaneous import mkdir

import torch
from PIL import Image
from random import random
import numpy as np

from mmkg_vrd.scene_graph_benchmark.scene_parser import SceneParser
from mmkg_vrd.scene_graph_benchmark.AttrRCNN import AttrRCNN


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and 
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]

    if isinstance(model, SceneParser):
        prediction_pred = prediction.prediction_pairs
        relations = prediction_pred.get_field("idx_pairs").tolist()
        relation_scores = prediction_pred.get_field("scores").tolist()
        predicates = prediction_pred.get_field("labels").tolist()
        prediction = prediction.predictions

    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()

    if isinstance(model, SceneParser):
        rt_box_list = []
        if 'attr_scores' in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist()}
                for box, cls, score, attr, attr_conf in
                zip(boxes, classes, scores, attr_labels, attr_scores)
            ]
        else:
            rt_box_list = [
                {"rect": box, "class": cls, "conf": score}
                for box, cls, score in
                zip(boxes, classes, scores)
            ]
        rt_relation_list = [{"subj_id": relation[0], "obj_id":relation[1], "class": predicate+1, "conf": score}
                for relation, predicate, score in
                zip(relations, predicates, relation_scores)]
        return {'objects': rt_box_list, 'relations':rt_relation_list}
    else:
        if 'attr_scores' in prediction.extra_fields:
            attr_scores = prediction.get_field("attr_scores")
            attr_labels = prediction.get_field("attr_labels")
            return [
                {"rect": box, "class": cls, "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist()}
                for box, cls, score, attr, attr_conf in
                zip(boxes, classes, scores, attr_labels, attr_scores)
            ]

        return [
            {"rect": box, "class": cls, "conf": score}
            for box, cls, score in
            zip(boxes, classes, scores)
        ]


def get_font_info(img_size):
    font = cv2.FONT_HERSHEY_SIMPLEX  # default font
    ref = (img_size[0] + img_size[1]) / 2
    font_scale = ref / 1000
    font_thickness = int(max(ref / 400, 1))
    return font, font_scale, font_thickness


def get_random_colors(labels):
    label_set = set(labels)
    label_counts = [(l, labels.count(l)) for l in label_set]
    label_counts = sorted(label_counts, key=lambda x: -x[1])
    label2color = {}
    # the most frequeny classes will get the gold colors
    gold_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for i, lab in enumerate(label_counts):
        if i < len(gold_colors):
            label2color[lab[0]] = gold_colors[i]
        else:
            label2color[lab[0]] = (random() * 255, random() * 255, 
                                   random() * 255)
    return label2color


def put_text(im, text, bottomleft=(0,100), color=(255,255,255), 
        font_scale=0.5, font_thickness=1):
    # function borrowed from quickdetection
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, text, bottomleft, font, font_scale, color,
            thickness=font_thickness)
    return cv2.getTextSize(text, font, font_scale, font_thickness)[0]


def draw_bb(im, all_rect, all_label, probs=None, color=None, 
        draw_label=True):
    '''
    function borrowed from quickdetection.
    all_rect: in xyxy mode
    all_label: list of class names
    probs: list of confidence scores, will show if given
    '''
    font, font_scale, font_thickness = get_font_info(im.shape[:2])

    dist_label = set(all_label)
    if color is None:
        label_to_color = {}
        gold_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),]
        color = label_to_color
        for l in dist_label:
            if l in color:
                continue
            if len(gold_colors) > 0:
                color[l] = gold_colors.pop()
    for i, l in enumerate(dist_label):
        if l in color:
            continue
        color[l] = (random() * 255., random() * 255, random() * 255)

    if type(all_rect) is list:
        assert len(all_rect) == len(all_label) 
    elif type(all_rect) is np.ndarray:
        assert all_rect.shape[0] == len(all_label)
        assert all_rect.shape[1] == 4
    else:
        assert False
    
    for i in range(len(all_label)):
        rect = all_rect[i]
        label = all_label[i]
        cv2.rectangle(im, (int(rect[0]), int(rect[1])), 
                (int(rect[2]), int(rect[3])), color[label], 
                thickness=font_thickness)
        if probs is not None:
            if draw_label:
                label_in_image = '{}-{:.2f}'.format(label, probs[i])
            else:
                label_in_image = '{:.2f}'.format(probs[i])
        else:
            if draw_label:
                label_in_image = '{}'.format(label)

        def gen_candidate():
            # above of top left
            yield int(rect[0]) + 2, int(rect[1]) - 4
            # below of bottom left
            yield int(rect[0]) + 2, int(rect[3]) + text_height + 2
        if draw_label or probs is not None:
            (_, text_height), _ = cv2.getTextSize(label_in_image, font,
                                                  font_scale, font_thickness)
            for text_left, text_bottom in gen_candidate():
                if 0 <= text_left < im.shape[1] - 12 and 12 < text_bottom  < im.shape[0]:
                    put_text(im, label_in_image, (text_left, text_bottom), color[label],
                            font_scale, font_thickness)
                    break


def get_text_image(img_size, text, bg_color=(0,0,0), text_color=(255,255,255)):
    # generate a blank image with centered text
    font, font_scale, font_thickness = get_font_info(img_size)
    img = np.ones((img_size[1], img_size[0], 3), np.uint8)
    img[:,:,0] = img[:,:,0] * bg_color[0]
    img[:,:,1] = img[:,:,1] * bg_color[1]
    img[:,:,2] = img[:,:,2] * bg_color[2]
    textsize = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, text, (textX, textY), font, font_scale, text_color, font_thickness)
    return img


def load_colormap_file(colormap_file):
    if not colormap_file:
        return 
    label2color = {}
    with open(colormap_file, 'r') as f:
        for line in f:
            item = line.strip().split('\t')
            label2color[item[0]] = [float(_) for _ in item[1].split(' ')]
    return label2color


def draw_rel(im, all_rel_subj_center, all_rel_obj_center, all_rel_label, probs=None, color=None, 
        draw_label=True):
    font, font_scale, font_thickness = get_font_info(im.shape[:2])

    dist_label = set(all_rel_label)
    if color is None:
        label_to_color = {}
        gold_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),]
        color = label_to_color
        for l in dist_label:
            if l in color:
                continue
            if len(gold_colors) > 0:
                color[l] = gold_colors.pop()
    for i, l in enumerate(dist_label):
        if l in color:
            continue
        color[l] = (random() * 255., random() * 255, random() * 255)
    
    for i in range(len(all_rel_label)):
        subj_center = all_rel_subj_center[i]
        obj_center = all_rel_obj_center[i]
        label = all_rel_label[i]
        cv2.line(im, (int(subj_center[0]), int(subj_center[1])), 
                (int(obj_center[0]), int(obj_center[1])), color[label], 
                thickness=font_thickness)
        if probs is not None:
            if draw_label:
                label_in_image = '{}-{:.2f}'.format(label, probs[i])
            else:
                label_in_image = '{:.2f}'.format(probs[i])
        else:
            if draw_label:
                label_in_image = '{}'.format(label)

        def gen_candidate():
            # above of top left
            rel_center = [(subj_center[0]+obj_center[0])/2, (subj_center[1]+obj_center[1])/2]
            yield int(rel_center[0]) + 2, int(rel_center[1]) - 4
            # below of bottom left
            yield int(rel_center[0]) + 2, int(rel_center[1]) + text_height + 2
        if draw_label or probs is not None:
            (_, text_height), _ = cv2.getTextSize(label_in_image, font,
                                                  font_scale, font_thickness)
            for text_left, text_bottom in gen_candidate():
                if 0 <= text_left < im.shape[1] - 12 and 12 < text_bottom  < im.shape[0]:
                    put_text(im, label_in_image, (text_left, text_bottom), color[label],
                            font_scale, font_thickness)
                    break


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden',
        'wood', 'orange', 'gray', 'grey', 'metal', 'pink', 'tall', 'long', 'dark', 'purple'
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {'blonde': 'blond'}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]
    

def get_opts(mode='vrd'):
    if mode == 'vrd':
        return [
            'MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED', False,
            "MODEL.META_ARCHITECTURE", "SceneParser",
            "MODEL.USE_FREQ_PRIOR", False,
            "MODEL.BACKBONE.CONV_BODY", "R-152-FPN",
            "MODEL.RESNETS.BACKBONE_OUT_CHANNELS", 256,
            "MODEL.RESNETS.STRIDE_IN_1X1", False,
            "MODEL.RESNETS.NUM_GROUPS", 32,
            "MODEL.RESNETS.WIDTH_PER_GROUP", 8,
            "MODEL.RPN.USE_FPN", True,
            "MODEL.RPN.ANCHOR_STRIDE", (4, 8, 16, 32, 64),
            "MODEL.RPN.PRE_NMS_TOP_N_TRAIN", 2000,
            "MODEL.RPN.PRE_NMS_TOP_N_TEST", 1000,
            "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN", 1000,
            "MODEL.RPN.FPN_POST_NMS_TOP_N_TEST", 1000,
            "MODEL.ROI_HEADS.USE_FPN", True,
            "MODEL.ROI_HEADS.POSITIVE_FRACTION", 0.5,
            "MODEL.ROI_HEADS.SCORE_THRESH", 0.05,
            "MODEL.ROI_HEADS.DETECTIONS_PER_IMG", 100,
            "MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG", 0,
            "MODEL.ROI_BOX_HEAD.NUM_CLASSES", 58,
            "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION", 7,
            "MODEL.ROI_BOX_HEAD.POOLER_SCALES", (0.25, 0.125, 0.0625, 0.03125),
            "MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO", 2,
            "MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR", "FPN2MLPFeatureExtractor",
            "MODEL.ROI_BOX_HEAD.PREDICTOR", "FPNPredictor",
            "MODEL.ATTRIBUTE_ON", False,
            "MODEL.RELATION_ON", True,
            "MODEL.ROI_RELATION_HEAD.DETECTOR_BOX_THRESHOLD", 0.05,
            "MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS", False,
            "MODEL.ROI_RELATION_HEAD.ALGORITHM", "sg_reldn",
            "MODEL.ROI_RELATION_HEAD.MODE", 'sgdet',
            "MODEL.ROI_RELATION_HEAD.USE_BIAS", False,
            "MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP", True,
            "MODEL.ROI_RELATION_HEAD.UPDATE_BOX_REG", False,
            "MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE", False,
            "MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR", False,
            "MODEL.ROI_RELATION_HEAD.SEPERATE_SO_FEATURE_EXTRACTOR", True,
            "MODEL.ROI_RELATION_HEAD.NUM_CLASSES", 10,
            "MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION", 7,
            "MODEL.ROI_RELATION_HEAD.POOLER_SCALES", (0.25, 0.125, 0.0625, 0.03125),
            "MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO", 2,
            "MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR", "FPN2MLPRelationFeatureExtractor",
            "MODEL.ROI_RELATION_HEAD.PREDICTOR", "FPNRelationPredictor",
            "MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.USE_FLAG", True,
            "MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG", 100,
            "MODEL.ROI_RELATION_HEAD.POSTPROCESS_METHOD", 'unconstrained',
            "INPUT.PIXEL_MEAN", [103.530, 116.280, 123.675],
            "DATASETS.FACTORY_TRAIN", ("OpenImagesVRDTSVDataset",),
            "DATASETS.FACTORY_TEST", ("OpenImagesVRDTSVDataset",),
            "DATASETS.TRAIN", ("openimages_v5c/vrd/train.vrd.challenge18.R152detector_pre_calculate.yaml",),
            "DATASETS.TEST", ("openimages_v5c/vrd/val.vrd.challenge18.R152detector_pre_calculate.yaml",),
            "DATALOADER.SIZE_DIVISIBILITY", 32,
            "DATALOADER.NUM_WORKERS", 0,
            "SOLVER.BASE_LR", 0.005,
            "SOLVER.WEIGHT_DECAY", 0.0001,
            "SOLVER.MAX_ITER", 162050,
            "SOLVER.STEPS", (97000, 129600),
            "SOLVER.IMS_PER_BATCH", 1,
            "SOLVER.CHECKPOINT_PERIOD", 20000,
            "TEST.IMS_PER_BATCH", 1,
            "TEST.SAVE_PREDICTIONS", True,
            "TEST.SAVE_RESULTS_TO_TSV", True,
            "TEST.TSV_SAVE_SUBSET", ['rect', 'class', 'conf', 'relations', 'relation_scores'],
            "TEST.GATHER_ON_CPU", True,
            "TEST.SKIP_PERFORMANCE_EVAL", False,
            "OUTPUT_DIR", "./exps/ji_relation_X152FPN_test",
            "DATA_DIR", "./mmkg_vrd/datasets",
            "DISTRIBUTED_BACKEND", 'gloo',
        ]
    raise RuntimeError(f'invalid mode {mode}')


class VRD:

    def __init__(
        self, 
        mode='vrd', 
        pretrained_ckpt=None, # 'custom_io/ckpt/RX152FPN_reldn_oi_best.pth'
        labelmap_file=None,
        freq_prior=None, # 'mmkg_vrd/datasets/openimages_v5c/vrd/vrd_frequency_prior_include_background.npy'
        ):
        self.mode = mode
        # self.config_file = str(Path(__file__).parent / 'sgg_configs' / 'vrd' / 'R152FPN_vrd_reldn.yaml')

        opts = get_opts(mode)
        if pretrained_ckpt is None or labelmap_file is None or freq_prior is None:
            print("Please read README.md from https://github.com/xu7yue/mmkg-vrd, and download these files.")
            exit()
        if pretrained_ckpt is not None:
            assert os.path.exists(pretrained_ckpt) is True
            opts.extend(['MODEL.WEIGHT', pretrained_ckpt])
        if labelmap_file is not None:
            assert os.path.exists(labelmap_file) is True
            opts.extend(['DATASETS.LABELMAP_FILE', labelmap_file])
        if freq_prior is not None:
            assert os.path.exists(freq_prior) is True
            opts.extend(['MODEL.FREQ_PRIOR', freq_prior])
        

        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        # cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()

        output_dir = cfg.OUTPUT_DIR
        mkdir(output_dir)

        if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
            self.model = SceneParser(cfg)
        elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)
        self.model.to('cuda')
        self.model.eval()

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.WEIGHT)

        # dataset labelmap is used to convert the prediction to class labels
        # dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
        dataset_labelmap_file = config_dataset_file(None, cfg.DATASETS.LABELMAP_FILE)
        assert dataset_labelmap_file
        dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
        self.dataset_labelmap = {int(val): key for key, val in dataset_allmap['label_to_idx'].items()}
        # visual_labelmap is used to select classes for visualization
        try:
            self.visual_labelmap = load_labelmap_file(labelmap_file)
        except:
            self.visual_labelmap = None

        if cfg.MODEL.ATTRIBUTE_ON and self.mode == 'attr':
            self.dataset_attr_labelmap = {
                int(val): key for key, val in
                dataset_allmap['attribute_to_idx'].items()}

        if cfg.MODEL.RELATION_ON and self.mode == 'vrd':
            self.dataset_relation_labelmap = {
                int(val): key for key, val in
                dataset_allmap['predicate_to_idx'].items()}

    def det_and_vis(self, img_file, save_file):
        assert op.isfile(img_file), \
            "Image: {} does not exist".format(img_file)
        
        transforms = build_transforms(cfg, is_train=False)
        cv2_img = cv2.imread(img_file)
        dets = detect_objects_on_single_image(self.model, transforms, cv2_img)

        if isinstance(self.model, SceneParser):
            rel_dets = dets['relations']
            dets = dets['objects']

        for obj in dets:
            obj["class"] = self.dataset_labelmap[obj["class"]]
        if self.visual_labelmap is not None:
            dets = [d for d in dets if d['class'] in self.visual_labelmap]
        if cfg.MODEL.ATTRIBUTE_ON and self.mode == 'attr':
            for obj in dets:
                obj["attr"], obj["attr_conf"] = postprocess_attr(self.dataset_attr_labelmap, obj["attr"], obj["attr_conf"])
        if cfg.MODEL.RELATION_ON and self.mode == 'vrd':
            for rel in rel_dets:
                rel['class'] = self.dataset_relation_labelmap[rel['class']]
                subj_rect = dets[rel['subj_id']]['rect']
                rel['subj_center'] = [(subj_rect[0]+subj_rect[2])/2, (subj_rect[1]+subj_rect[3])/2]
                obj_rect = dets[rel['obj_id']]['rect']
                rel['obj_center'] = [(obj_rect[0]+obj_rect[2])/2, (obj_rect[1]+obj_rect[3])/2]

            rects = [d["rect"] for d in dets]

        scores = [d["conf"] for d in dets]
        if cfg.MODEL.ATTRIBUTE_ON and self.mode == 'attr':
            attr_labels = [','.join(d["attr"]) for d in dets]
            attr_scores = [d["attr_conf"] for d in dets]
            labels = [attr_label+' '+d["class"]
                      for d, attr_label in zip(dets, attr_labels)]
        else:
            labels = [d["class"] for d in dets]

        draw_bb(cv2_img, rects, labels, scores)

        if cfg.MODEL.RELATION_ON and self.mode == 'vrd':
            rel_subj_centers = [r['subj_center'] for r in rel_dets]
            rel_obj_centers = [r['obj_center'] for r in rel_dets]
            rel_scores = [r['conf'] for r in rel_dets]
            rel_labels = [r['class'] for r in rel_dets]
            draw_rel(cv2_img, rel_subj_centers, rel_obj_centers, rel_labels, rel_scores)

        if not save_file:
            save_file = op.splitext(img_file)[0] + ".detect.jpg"
        else:
            save_file = save_file
        cv2.imwrite(save_file, cv2_img)
        print("save results to: {}".format(save_file))

        # save results in text
        if cfg.MODEL.ATTRIBUTE_ON and self.mode == 'attr':
            result_str = ""
            for label, score, attr_score in zip(labels, scores, attr_scores):
                result_str += label+'\n'
                result_str += ','.join([str(conf) for conf in attr_score])
                result_str += '\t'+str(score)+'\n'
            text_save_file = op.splitext(save_file)[0] + '.txt'
            with open(text_save_file, "w") as fid:
                fid.write(result_str)

if __name__ == "__main__":
    vrd = VRD(
        pretrained_ckpt='custom_io/ckpt/RX152FPN_reldn_oi_best.pth',
        labelmap_file='custom_io/ji_vrd_labelmap.json', 
        freq_prior='custom_io/vrd_frequency_prior_include_background.npy', 
    )
    vrd.det_and_vis(
        img_file='custom_io/imgs/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg',
        save_file='custom_io/out/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg')
    print('all done')
