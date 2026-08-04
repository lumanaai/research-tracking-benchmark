import argparse
import os
import json
import cv2
import numpy as np
import math
import time
import torch
from ...pytorchocr.base_ocr_v20 import BaseOCRV20
#import .pytorchocr_utility as utility
from . import pytorchocr_utility as utility

from ...pytorchocr.postprocess import build_post_process
from ...pytorchocr.utils.utility import get_image_file_list, check_and_read_gif


class ModelBuilder:
    def __init__(self, args):
        self.args = args
        self.rec_algorithm = args.rec_algorithm
        self.use_gpu = torch.cuda.is_available() and args.use_gpu
        if self.use_gpu:
            self.device = args.device
        self.use_half = args.use_half

        # Post-processing parameters
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        # Adjust postprocess_params based on rec_algorithm
        if self.rec_algorithm == "SRN":
            postprocess_params['name'] = 'SRNLabelDecode'
        elif self.rec_algorithm == "RARE":
            postprocess_params['name'] = 'AttnLabelDecode'
        elif self.rec_algorithm == 'NRTR':
            postprocess_params['name'] = 'NRTRLabelDecode'
        elif self.rec_algorithm == "SAR":
            postprocess_params['name'] = 'SARLabelDecode'
        elif self.rec_algorithm == 'ViTSTR':
            postprocess_params['name'] = 'ViTSTRLabelDecode'
        elif self.rec_algorithm == "CAN":
            postprocess_params['name'] = 'CANLabelDecode'
        elif self.rec_algorithm == 'RFL':
            postprocess_params = {
                'name': 'RFLLabelDecode',
                "character_dict_path": None,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)
        char_num = len(getattr(self.postprocess_op, 'character'))

        # Build model
        self.net = self.build_model(char_num)

    def build_model(self, char_num):
        network_config = utility.AnalysisConfig(self.args.rec_model_path, self.args.rec_yaml_path, char_num)
        weights = self.read_pytorch_weights(self.args.rec_model_path)

        self.out_channels = self.get_out_channels(weights)
        if self.rec_algorithm == 'NRTR':
            self.out_channels = list(weights.values())[-1].numpy().shape[0]
        elif self.rec_algorithm == 'SAR':
            self.out_channels = list(weights.values())[-3].numpy().shape[0]

        kwargs = {'out_channels': self.out_channels}
        model = BaseOCRV20(network_config, **kwargs)
        model.load_state_dict(weights)
        model.net.eval()
        if self.use_gpu:
            model.net.to(f'{self.device}')
        if self.use_half:
            model.net.half()
        return model.net

    def read_pytorch_weights(self, weights_path):
        # Read the PyTorch weights from the given path
        weights = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in weights:
            weights = weights['state_dict']
        return weights

    def get_out_channels(self, weights):
        # Get output channels based on the last layer's weight shape
        for k in weights.keys():
            if 'out' in k:
                out_channels = weights[k].shape[0]
                return out_channels
        # Default to last weight's shape
        return list(weights.values())[-1].shape[0]


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_algorithm = args.rec_algorithm
        self.limited_max_width = args.limited_max_width
        self.limited_min_width = args.limited_min_width
        self.max_text_length = args.max_text_length
        self.inverse = args.rec_image_inverse

    def preprocess(self, img_list):
        preprocessed_data_np = []

        for img in img_list:
            if self.rec_algorithm == "SAR":
                norm_img, _, _, valid_ratio = self.resize_norm_img_sar(img, self.rec_image_shape)
                preprocessed_data_np.append((norm_img, valid_ratio))
            elif self.rec_algorithm == "SVTR":
                norm_img = self.resize_norm_img_svtr(img, self.rec_image_shape)
                preprocessed_data_np.append(norm_img)
            elif self.rec_algorithm == "SRN":
                norm_img_tuple = self.process_image_srn(img, self.rec_image_shape, 8, self.max_text_length)
                preprocessed_data_np.append(norm_img_tuple)
            elif self.rec_algorithm == "CAN":
                norm_img = self.norm_img_can(img)
                norm_image_mask = np.ones(norm_img.shape, dtype='float32')
                word_label = np.ones([1, 36], dtype='int64')
                preprocessed_data_np.append((norm_img, norm_image_mask, word_label))
            else:
                norm_img = self.resize_norm_img(img)
                preprocessed_data_np.append(norm_img)

        return preprocessed_data_np
    

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio =  imgW / imgH
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32').transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im

    def resize_norm_img_svtr(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32').transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        img_black = np.zeros((imgH, imgW))
        im_hei, im_wid = img.shape[:2]
        scale_w = min(max((im_wid // im_hei), 1), 3)
        img_new = cv2.resize(img, (imgH * scale_w, imgH))
        img_np = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
        img_black[:, :img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]
        return img_black.transpose((2, 0, 1)).astype(np.float32)

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2 = self.srn_other_inputs(image_shape, num_heads, max_text_length)
        return (norm_img[np.newaxis, :], encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):
        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))
        encoder_word_pos = np.arange(feature_dim).reshape((feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.arange(max_text_length).reshape((max_text_length, 1)).astype('int64')
        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = (np.triu(gsrm_attn_bias_data, 1) * -1e9).astype('float32')
        gsrm_slf_attn_bias2 = (np.tril(gsrm_attn_bias_data, -1) * -1e9).astype('float32')
        gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1, (1, num_heads, 1, 1))
        gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2, (1, num_heads, 1, 1))
        return [encoder_word_pos[np.newaxis, :], gsrm_word_pos[np.newaxis, :], gsrm_slf_attn_bias1, gsrm_slf_attn_bias2]

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape[0], image_shape[1], 10, 2000  # Use default min and max widths
        h, w = img.shape[:2]
        valid_ratio = 1.0
        width_divisor = int(1 / width_downsample_ratio)
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        resize_w = max(imgW_min, resize_w)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        resize_w = min(imgW_max, resize_w)
        valid_ratio = min(1.0, resize_w / imgW_max)
        resized_image = cv2.resize(img, (resize_w, imgH)).astype('float32')
        if imgC == 1:
            resized_image = resized_image / 255.0
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose(2, 0, 1) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, :resize_w] = resized_image
        return padding_im, resized_image.shape, padding_im.shape, valid_ratio

    def norm_img_can(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.inverse:
            img = 255 - img
        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape
        if h < imgH or w < imgW:
            padding_h = max(imgH - h, 0)
            padding_w = max(imgW - w, 0)
            img = np.pad(img, ((0, padding_h), (0, padding_w)), 'constant', constant_values=255)
        img = np.expand_dims(img, 0).astype('float32') / 255.0
        return img



class InferenceEngine(torch.nn.Module):
    def __init__(self, net, args):
        super(InferenceEngine, self).__init__()
        self.net = net
        self.args = args
        self.rec_algorithm = args.rec_algorithm

        self.use_gpu = torch.cuda.is_available() and args.use_gpu
        self.use_half = args.use_half
        self.device = args.device if self.use_gpu else 'cpu'

    def forward(self, inp):
            with torch.no_grad():
                if self.use_half:
                    inp = inp.half()
                if self.use_gpu:
                    inp = inp.to(f'{self.device}')
                preds = self.net(inp)
            return preds



class Postprocessor:
    def __init__(self, args):
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        # Adjust postprocess_params based on rec_algorithm
        if args.rec_algorithm == "SRN":
            postprocess_params['name'] = 'SRNLabelDecode'
        elif args.rec_algorithm == "RARE":
            postprocess_params['name'] = 'AttnLabelDecode'
        elif args.rec_algorithm == 'NRTR':
            postprocess_params['name'] = 'NRTRLabelDecode'
        elif args.rec_algorithm == "SAR":
            postprocess_params['name'] = 'SARLabelDecode'
        elif args.rec_algorithm == 'ViTSTR':
            postprocess_params['name'] = 'ViTSTRLabelDecode'
        elif args.rec_algorithm == "CAN":
            postprocess_params['name'] = 'CANLabelDecode'
        elif args.rec_algorithm == 'RFL':
            postprocess_params = {
                'name': 'RFLLabelDecode',
                "character_dict_path": None,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)

    def clean_license_plate(self, ocr_result):
        return ''.join(c for c in ocr_result if c.isalnum())

    def postprocess(self, preds):
        rec_result = self.postprocess_op(preds)
        
        for rec in rec_result:
            rec = (self.clean_license_plate(rec[0]), rec[1])
        return rec_result


def warmup(preprocessor, inference_engine, args, iterations = 20, warmup_batch_size = 8):
    print("Starting warmup...")
    shape = [int(dim) for dim in args.rec_image_shape.split(',')][::-1]

    warmup_img_list = [np.random.randint(0, 255, shape).astype(np.uint8) for _ in range(warmup_batch_size)]

    preprocessed_data = preprocessor.preprocess(warmup_img_list)
    # Assuming the warmup just runs inference without caring about the outputs
    for _ in range(iterations):
        input = np.concatenate([data[np.newaxis, :] for data in preprocessed_data])
        input = torch.from_numpy(input)
        inference_engine(input)
    print("Warmup completed.")


