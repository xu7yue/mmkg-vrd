# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 

import cv2
import os.path as op
import argparse
import json

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


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--img_file", metavar="FILE", help="image path")
    parser.add_argument("--labelmap_file", metavar="FILE",
                        help="labelmap file to select classes for visualizatioin")
    parser.add_argument("--save_file", required=False, type=str, default=None,
                        help="filename to save the proceed image")
    parser.add_argument("--visualize_attr", action="store_true",
                        help="visualize the object attributes")
    parser.add_argument("--visualize_relation", action="store_true",
                        help="visualize the relationships")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    assert op.isfile(args.img_file), \
        "Image: {} does not exist".format(args.img_file)

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(cfg.DATA_DIR,
                                                cfg.DATASETS.LABELMAP_FILE)
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, 'r'))
    dataset_labelmap = {int(val): key
                        for key, val in dataset_allmap['label_to_idx'].items()}
    # visual_labelmap is used to select classes for visualization
    try:
        visual_labelmap = load_labelmap_file(args.labelmap_file)
    except:
        visual_labelmap = None

    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        dataset_attr_labelmap = {
            int(val): key for key, val in
            dataset_allmap['attribute_to_idx'].items()}
    
    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        dataset_relation_labelmap = {
            int(val): key for key, val in
            dataset_allmap['predicate_to_idx'].items()}

    transforms = build_transforms(cfg, is_train=False)
    cv2_img = cv2.imread(args.img_file)
    dets = detect_objects_on_single_image(model, transforms, cv2_img)

    if isinstance(model, SceneParser):
        rel_dets = dets['relations']
        dets = dets['objects']

    for obj in dets:
        obj["class"] = dataset_labelmap[obj["class"]]
    if visual_labelmap is not None:
        dets = [d for d in dets if d['class'] in visual_labelmap]
    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        for obj in dets:
            obj["attr"], obj["attr_conf"] = postprocess_attr(dataset_attr_labelmap, obj["attr"], obj["attr_conf"])
    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        for rel in rel_dets:
            rel['class'] = dataset_relation_labelmap[rel['class']]
            subj_rect = dets[rel['subj_id']]['rect']
            rel['subj_center'] = [(subj_rect[0]+subj_rect[2])/2, (subj_rect[1]+subj_rect[3])/2]
            obj_rect = dets[rel['obj_id']]['rect']
            rel['obj_center'] = [(obj_rect[0]+obj_rect[2])/2, (obj_rect[1]+obj_rect[3])/2]


    rects = [d["rect"] for d in dets]
    scores = [d["conf"] for d in dets]
    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        attr_labels = [','.join(d["attr"]) for d in dets]
        attr_scores = [d["attr_conf"] for d in dets]
        labels = [attr_label+' '+d["class"]
                  for d, attr_label in zip(dets, attr_labels)]
    else:
        labels = [d["class"] for d in dets]

    draw_bb(cv2_img, rects, labels, scores)

    if cfg.MODEL.RELATION_ON and args.visualize_relation:
        rel_subj_centers = [r['subj_center'] for r in rel_dets]
        rel_obj_centers = [r['obj_center'] for r in rel_dets]
        rel_scores = [r['conf'] for r in rel_dets]
        rel_labels = [r['class'] for r in rel_dets]
        draw_rel(cv2_img, rel_subj_centers, rel_obj_centers, rel_labels, rel_scores)

    if not args.save_file:
        save_file = op.splitext(args.img_file)[0] + ".detect.jpg"
    else:
        save_file = args.save_file
    cv2.imwrite(save_file, cv2_img)
    print("save results to: {}".format(save_file))

    # save results in text
    if cfg.MODEL.ATTRIBUTE_ON and args.visualize_attr:
        result_str = ""
        for label, score, attr_score in zip(labels, scores, attr_scores):
            result_str += label+'\n'
            result_str += ','.join([str(conf) for conf in attr_score])
            result_str += '\t'+str(score)+'\n'
        text_save_file = op.splitext(save_file)[0] + '.txt'
        with open(text_save_file, "w") as fid:
            fid.write(result_str)


if __name__ == "__main__":
    main()
