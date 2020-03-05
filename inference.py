# coding: utf-8

import argparse
import os
import cv2
import time
import json
import glob
import numpy as np

import torch
from utils import restore_rectangle
from models import East, East_Resnet18
import lanms


def get_images(data_dir):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(data_dir, '*.{}'.format(ext))))
    return files


def resize_image(im, max_side_len=768):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def bf_resize_image(im, input_size=(672, 480)):
    # assert input_size % 32 == 0

    h, w, _ = im.shape

    im = cv2.resize(im, input_size)
    resize_h, resize_w, _ = im.shape

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def preprocess(im, input_size):
    '''
    load image and preprocess
    '''
    if isinstance(input_size, int):
        im_resized, (ratio_h, ratio_w) = resize_image(im, input_size)
    elif len(input_size) == 1:
        im_resized, (ratio_h, ratio_w) = resize_image(im, input_size[0])
    else:
        im_resized, (ratio_h, ratio_w) = bf_resize_image(im, tuple(input_size))
    im_resized = im_resized.astype(np.float32)
    im_resized = im_resized.transpose(2, 0, 1)
    im_resized = torch.from_numpy(im_resized)
    im_resized = im_resized.cuda()
    im_resized = im_resized.unsqueeze(0)

    return im_resized, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thresh=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshold for score map
    :param box_thresh: threshold for boxes
    :param nms_thresh: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh) # score > 阈值的indices, shape=(N, 2)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4,
                                          geo_map[xy_text[:, 0],
                                          xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thresh)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map
    # this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def predict(model, image_list, result_file, input_size, thresholds=(0.8, 0.1, 0.2), draw_dir=None):
    '''
    thresholds: tuple. size=3
        (score_map_thresh: threshhold for score map
         box_thresh: threshhold for boxes
         nms_thres: threshold for nms)
    '''
    model.eval()

    im_fn_list = image_list
    # im_fn_list = get_images(data_dir)
    f = open(result_file, 'w')
    with torch.no_grad():
        model_time = 0.0
        count = 0
        for idx, im_fn in enumerate(im_fn_list):
            im = cv2.imread(im_fn)
            im_h, im_w, _ = im.shape
            if im is None:
                print('invalid image: %s' % im_fn)
                continue

            im = im[:, :, ::-1] #bgr to rgb
            im_resized, (ratio_h, ratio_w) = preprocess(im, input_size)
            if im_resized is None:
                continue

            timer = {'net': 0, 'restore': 0, 'nms':0}
            start_time = time.time()
            f_score, f_geometry = model(im_resized)
            timer['net'] = time.time() - start_time
            print('Predict %d Image, Model Time: %.3f ms' % (idx, timer['net']*1000))
            count += 1
            model_time += timer['net'] * 1000

            # post process
            f_score = f_score.permute(0, 2, 3, 1)
            f_geometry = f_geometry.permute(0, 2, 3, 1)
            f_score = f_score.data.cpu().numpy()
            f_geometry = f_geometry.data.cpu().numpy()

            # restore
            boxes, timer = detect(score_map=f_score, geo_map=f_geometry, timer=timer,
                                  score_map_thresh=thresholds[0],
                                  box_thresh=thresholds[1],
                                  nms_thresh=thresholds[2])
            print('Predict %d Image, Restore Time: %.3f ms' % (idx, timer['restore']*1000))
            print('Predict %d Image, NMS Time: %.3f ms' % (idx, timer['nms']*1000))

            # save boxes
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
            
            if draw_dir:
                img_outdir = draw_dir
                os.makedirs(img_outdir, exist_ok=True)
                draw(im[:, :, ::-1], boxes, os.path.join(img_outdir, os.path.basename(im_fn)))

            res = {}
            res['url'] = im_fn
            res['texts'] = []

            if boxes is not None:
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    item = {}
                    item['text'] = 'abc'
                    item['bboxes'] = [box[0, 0], box[0, 1],
                                      box[1, 0], box[1, 1],
                                      box[2, 0], box[2, 1],
                                      box[3, 0], box[3, 1]]
                    item['bboxes'] = list(map(lambda x: int(x.tolist()), item['bboxes']))
                    res['texts'].append(item)

            f.write('%s\n'%str(json.dumps(res)))

    f.close()
    print('Avg Model Time: {}'.format(model_time/count))


def draw(img, polys, dst):
    img = img.copy()
    if polys is not None:
        for poly in polys:
            poly = poly.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [poly], True, (255, 255, 0), thickness=2)
    cv2.imwrite(dst, img)


if __name__ == '__main__':
    args = argparse.ArgumentParser('EAST detect')
    args.add_argument('model', metavar='PTH', type=str,
        help='model path')
    args.add_argument('datadir', metavar='DIR', type=str,
        help='image dir')
    args.add_argument('output', metavar='RES', type=str,
        help='result file')
    args.add_argument('--draw', type=str,
        help='draw dir')
    args.add_argument('--backbone', type=str, choices=['resnet50', 'resnet18'],
        default='resnet50',
        help='backbone, resnet50 or resnet18, default=resnet50')
    args.add_argument('--text_scale', type=int, default=512,
        help='text scale, default=512')
    args.add_argument('--input_size', nargs='+', type=int, default=768,
        help='input image size, int(max side) or tuple(w, h), default=768.')
    args.add_argument('--score_map_thresh', type=float, default=0.8,
        help='threshold for score map, default=0.8')
    args.add_argument('--box_thresh', type=float, default=0.1,
        help='threshold for boxes, default=0.1')
    args.add_argument('--nms_thresh', type=float, default=0.2,
        help='nms threshold, default=0.2')

    args = args.parse_args()

    state = torch.load(args.model)
    if args.backbone == 'resnet18':
        net = East_Resnet18
    else:
        net = East
    model = net(text_scale=args.text_scale)
    model = model.cuda()
    # model.load_state_dict(checkpoint['model_state_dict'])
    state_dict = dict()
    for k, v in state['model_state_dict'].items():
        state_dict[k.replace('module.','')] = v
    model.load_state_dict(state_dict)

    if os.path.isdir(args.datadir):
        image_list = get_images(args.datadir)
    elif args.datadir.endswith('.json'):
        image_list = []
        with open(args.datadir, 'r') as f:
            for line in f:
                url = json.loads(line.strip())['url']
                if (url.startswith('qiniu:///')):
                    url = url.replace('qiniu:///', '/workspace/mnt/bucket/', 1)
                image_list.append(url)
    predict(model, image_list, args.output, input_size=args.input_size,
            thresholds=(args.score_map_thresh, args.box_thresh, args.nms_thresh),
            draw_dir=args.draw)
