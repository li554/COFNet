import argparse
import os
import random

import torch.cuda

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

import cv2
import numpy as np
from deep_utils import Box, split_extension

from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from utils.torch_utils import select_device

def get_res_img2(heat, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    heat.append(n_heatmat)
    return res_img, heat


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # n_heatmat = (Box.fill_outer_box(heatmap, bbox, value=0) / 255).astype(np.float32)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = cv2.addWeighted(res_img, 0.7, n_heatmat, 0.3, 0)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    # cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    # res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    # res_img = Box.put_text(res_img, cls_name, (x1 - 3, y1))
    # res_img = Box.put_text(res_img, str(round((float(cls_name)+0.40), 2)), (x1-3, y1))
    return res_img


def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img


def main(img_vis_path, img_ir_path):
    img_vis, img_ir = cv2.imread(img_vis_path), cv2.imread(img_ir_path)
    # preprocess the images
    torch_img_vis, torch_img_ir = model.preprocessing(img_vis[..., ::-1], img_ir[..., ::-1])
    result = torch_img_vis.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr

    ori_img = result.copy()
    images = []
    if args.method == 'gradcam':
        for layer in args.target_layer:
            saliency_method = YOLOV5GradCAM(model=model, layer_name=layer, img_size=input_size)
            tic = time.time()
            masks, logits, [boxes, _, class_names, confs] = saliency_method(torch_img_vis, torch_img_ir)
            print("total time:", round(time.time() - tic, 4))
            res_img = result.copy()
            res_img = res_img / 255
            heat = []
            for i, mask in enumerate(masks):
                bbox = boxes[0][i]
                mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
                    np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                n_heatmat = (heatmap / 255).astype(np.float32)
                heat.append(n_heatmat)

            if len(heat) != 0:
                heat_all = heat[0]
                for h in heat[1:]:
                    heat_all += h
                heat_avg = heat_all / len(heat)
                res_img = cv2.addWeighted(res_img, 0.3, heat_avg, 0.7, 0)

            res_img = (res_img / res_img.max())
            cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
            heat_map = cv2.imread('temp.jpg')
            final_image = heat_map
            images.append(final_image)
            # save the images
            suffix = '-res-' + layer
            img_name = split_extension(os.path.split(img_vis_path)[-1], suffix=suffix)
            output_path = f'{args.output_dir}/{img_name}'
            os.makedirs(args.output_dir, exist_ok=True)
            print(f'[INFO] Saving the final image at {output_path}')
            cv2.imwrite(output_path, final_image)

        img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_avg')
        output_path = f'{args.output_dir}/{img_name}'
        img_all = images[0].astype(np.uint16)
        for img in images[1:]:
            img_all += img
        img_avg = img_all / len(images)
        cv2.imwrite(output_path, img_avg.astype(np.uint8))
        img_name = split_extension(os.path.split(img_vis_path)[-1], suffix='_')
        cv2.imwrite(f'{args.output_dir}/{img_name}', ori_img.astype(np.uint8))
        del img_vis, img_ir, torch_img_vis, torch_img_ir


if __name__ == '__main__':
    """
    'model_30_cv1_act', 'model_30_cv2_act', 'model_30_cv3_act',
    'model_33_cv1_act', 'model_33_cv2_act', 'model_33_cv3_act',
    """
    # target = [
    #     'model_9_cv1_act', 'model_19_cv1_act',
    #     'model_30_cv1_act', 'model_30_cv2_act', 'model_30_cv3_act',
    #     'model_30_cv1_act', 'model_33_cv2_act', 'model_33_cv3_act']

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=f'./models/transformer/yolov5l_MaskFusion_mask_FLIR_wb.yaml',
                        help='model.yaml')
    parser.add_argument('--model-path', type=str, default=f"runs/train/LLVIP-mse-vmask/LLVIP_fu_avg_pool/weights/best.pt", help='Path to the model')
    parser.add_argument('--dataset', type=str, default='FLIR', help='dataset')
    parser.add_argument('--source1', type=str, default=f'data/FLIR-align-3class/visible/test',
                        # 'data/visualization/visible/'
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source2', type=str, default=f'data/FLIR-align-3class/infrared/test',
                        # 'data/visualization/infrared/'
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output-dir', type=str, default=f'runs/visualization/cams/FLIR/Ours2/', help='output dir')
    parser.add_argument('--name', type=str, default='Ours', help='name')
    parser.add_argument('--img-size', type=int, default=(640, 640), help="input image size")
    parser.add_argument('--target-layer', type=str, default=target,
                        help='The layer hierarchical address to which gradcam will applied,'
                             ' the names should be separated by underline')
    parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
    parser.add_argument('--device', type=str, default='0', help='cuda or cpu')
    parser.add_argument('--names', type=str, default=None,
                        help='The name of the classes. The default is set to None and is set to coco classes. '
                             'Provide your custom names as follow: object1,object2,object3')
    args = parser.parse_args()

    if args.dataset == 'FLIR':
        args.source1 = 'data/FLIR-align-3class/visible/test'
        args.source2 = 'data/FLIR-align-3class/infrared/test'
        args.output_dir = f'runs/visualization/cams/FLIR/{args.name}/'
        args.img_size = (640, 640)
    elif args.dataset == 'LLVIP':
        args.source1 = 'data/LLVIP/visible/test'
        args.source2 = 'data/LLVIP/infrared/test'
        args.output_dir = f'runs/visualization/cams/LLVIP/{args.name}/'
        args.img_size = (640, 640)
    elif args.dataset == 'kaist':
        args.source1 = 'data/kaist/visible/test'
        args.source2 = 'data/kaist/infrared/test'
        args.output_dir = f'runs/visualization/cams/kaist/{args.name}/'
        args.img_size = (640, 640)
    elif args.dataset == 'VEDAI':
        args.source1 = 'data/VEDAI_1024/fold01/visible/test'
        args.source2 = 'data/VEDAI_1024/fold01/infrared/test'
        args.output_dir = f'runs/visualization/cams/VEDAI/{args.name}/'
        args.img_size = (1024, 1024)

    device = select_device(args.device, batch_size=1)
    input_size = args.img_size
    # load model
    print('[INFO] Loading the model')
    model = YOLOV5TorchObjectDetector(args.cfg, args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","),
                                      confidence=0.3)
    random.seed(0)
    if os.path.isdir(args.source1):
        img_vis_list = os.listdir(args.source1)
        ## 按最大间隔取10张图片
        indexes = np.linspace(0, len(img_vis_list) - 1, 10, dtype=int)
        # 找到img_vis_list中名字包含9397的图片
        # indexes = [i for i in range(len(img_vis_list)) if '9397' in img_vis_list[i]]
        for index in indexes:
            item = img_vis_list[index + 5 if index + 5 < len(img_vis_list) else index]
            img_vis_path = os.path.join(args.source1, item)
            if (args.source1 == 'data/FLIR-align-3class/visible/test'
                    or args.source1 == 'data/visualization/visible/'):
                new_item = item[:-4] + '.jpeg'
                img_ir_path = os.path.join(args.source2, new_item)
            else:
                img_ir_path = os.path.join(args.source2, item)
            main(img_vis_path, img_ir_path)
            torch.cuda.empty_cache()
    else:
        main(args.source1, args.source2)
