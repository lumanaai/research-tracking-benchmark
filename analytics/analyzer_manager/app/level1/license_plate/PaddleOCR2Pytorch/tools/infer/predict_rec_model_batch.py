import argparse
import os
import sys
import json
import cv2
import numpy as np
import math
import time
import torch
from PIL import Image
from pytorchocr.base_ocr_v20 import BaseOCRV20
import tools.infer.pytorchocr_utility as utility
from pytorchocr.postprocess import build_post_process
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif

# Configuration Namespace
config = argparse.Namespace(
    image_dir='/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/lp/',
    limited_max_width=1280,
    limited_min_width=16,
    max_text_length=10,
    output_json_path='/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/finetuned-v4-owl_crops-torch.json',
    rec_algorithm='CRNN',
    rec_batch_num=10,
    rec_char_dict_path='./ppocr/utils/en_dict.txt',
    rec_char_type='en',
    rec_image_shape='3,48,320',
    rec_model_path='./en_ptocr_v4_rec_infer.pth',
    rec_yaml_path='./configs/rec/PP-OCRv4/en_PP-OCRv4_rec_LPR.yml',
    scales=[8, 16, 32],
    show_log=False,
    use_space_char=True,
    use_gpu=True,
    device=1,  # Ensure 'device' is specified for GPU usage
    use_half=True,
)

class TextRecognizer(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.max_text_length = args.max_text_length

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
            self.inverse = args.rec_image_inverse
            postprocess_params['name'] = 'CANLabelDecode'
        elif self.rec_algorithm == 'RFL':
            postprocess_params = {
                'name': 'RFLLabelDecode',
                "character_dict_path": None,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)

        use_gpu = args.use_gpu
        self.use_gpu = torch.cuda.is_available() and use_gpu
        if self.use_gpu:
            self.device = args.device
        self.use_half = args.use_half

        self.limited_max_width = args.limited_max_width
        self.limited_min_width = args.limited_min_width

        self.weights_path = args.rec_model_path
        self.yaml_path = args.rec_yaml_path

        char_num = len(getattr(self.postprocess_op, 'character'))
        network_config = utility.AnalysisConfig(self.weights_path, self.yaml_path, char_num)

        weights = self.read_pytorch_weights(self.weights_path)

        self.out_channels = self.get_out_channels(weights)
        if self.rec_algorithm == 'NRTR':
            self.out_channels = list(weights.values())[-1].numpy().shape[0]
        elif self.rec_algorithm == 'SAR':
            self.out_channels = list(weights.values())[-3].numpy().shape[0]

        kwargs['out_channels'] = self.out_channels
        super(TextRecognizer, self).__init__(network_config, **kwargs)

        self.load_state_dict(weights)
        self.net.eval()
        if self.use_gpu:
            self.net.to(f'cuda:{self.device}')
        if self.use_half:
            self.net.half()

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm in ['NRTR', 'ViTSTR']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_pil = Image.fromarray(np.uint8(img))
            img = image_pil.resize((imgW, imgH), Image.BICUBIC if self.rec_algorithm == 'ViTSTR' else Image.ANTIALIAS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            norm_img = norm_img.astype(np.float32) / (255. if self.rec_algorithm == 'ViTSTR' else 128.) - (0 if self.rec_algorithm == 'ViTSTR' else 1.)
            return norm_img
        elif self.rec_algorithm == 'RFL':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype('float32') / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image

        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((imgH * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
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

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2 = self.srn_other_inputs(image_shape, num_heads, max_text_length)
        return (norm_img[np.newaxis, :], encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2)

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h, w = img.shape[:2]
        valid_ratio = 1.0
        width_divisor = int(1 / width_downsample_ratio)
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        resize_w = max(imgW_min or resize_w, resize_w)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        resize_w = min(imgW_max or resize_w, resize_w)
        valid_ratio = min(1.0, resize_w / (imgW_max or resize_w))
        resized_image = cv2.resize(img, (resize_w, imgH)).astype('float32')
        if imgC == 1:
            resized_image = resized_image / 255.0
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose(2, 0, 1) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max or resize_w), dtype=np.float32)
        padding_im[:, :, :resize_w] = resized_image
        return padding_im, resized_image.shape, padding_im.shape, valid_ratio

    def norm_img_can(self, img, max_wh_ratio):
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

    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = np.array([img.shape[1] / float(img.shape[0]) for img in img_list])

        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = max(width_list[indices[beg_img_no:end_img_no]])

            # Initialize variables based on rec_algorithm
            if self.rec_algorithm == "SAR":
                valid_ratios = []
            elif self.rec_algorithm == "SRN":
                encoder_word_pos_list = []
                gsrm_word_pos_list = []
                gsrm_slf_attn_bias1_list = []
                gsrm_slf_attn_bias2_list = []
            elif self.rec_algorithm == "CAN":
                norm_img_mask_batch = []
                word_label_list = []

            for ino in range(beg_img_no, end_img_no):
                idx = indices[ino]
                img = img_list[idx]
                if self.rec_algorithm == "SAR":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(img, self.rec_image_shape)
                    valid_ratios.append(np.expand_dims(valid_ratio, axis=0))
                elif self.rec_algorithm == "SVTR":
                    norm_img = self.resize_norm_img_svtr(img, self.rec_image_shape)
                elif self.rec_algorithm == "SRN":
                    norm_img_tuple = self.process_image_srn(img, self.rec_image_shape, 8, self.max_text_length)
                    norm_img = norm_img_tuple[0]
                    encoder_word_pos_list.append(norm_img_tuple[1])
                    gsrm_word_pos_list.append(norm_img_tuple[2])
                    gsrm_slf_attn_bias1_list.append(norm_img_tuple[3])
                    gsrm_slf_attn_bias2_list.append(norm_img_tuple[4])
                elif self.rec_algorithm == "CAN":
                    norm_img = self.norm_img_can(img, max_wh_ratio)
                    norm_image_mask = np.ones(norm_img.shape, dtype='float32')
                    word_label = np.ones([1, 36], dtype='int64')
                    norm_img_mask_batch.append(norm_image_mask)
                    word_label_list.append(word_label)
                else:
                    norm_img = self.resize_norm_img(img, max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(gsrm_slf_attn_bias2_list)
                with torch.no_grad():
                    inp = torch.from_numpy(norm_img_batch)
                    encoder_word_pos_inp = torch.from_numpy(encoder_word_pos_list)
                    gsrm_word_pos_inp = torch.from_numpy(gsrm_word_pos_list)
                    gsrm_slf_attn_bias1_inp = torch.from_numpy(gsrm_slf_attn_bias1_list)
                    gsrm_slf_attn_bias2_inp = torch.from_numpy(gsrm_slf_attn_bias2_list)
                    if self.use_gpu:
                        inp = inp.to(f'cuda:{self.device}')
                        encoder_word_pos_inp = encoder_word_pos_inp.to(f'cuda:{self.device}')
                        gsrm_word_pos_inp = gsrm_word_pos_inp.to(f'cuda:{self.device}')
                        gsrm_slf_attn_bias1_inp = gsrm_slf_attn_bias1_inp.to(f'cuda:{self.device}')
                        gsrm_slf_attn_bias2_inp = gsrm_slf_attn_bias2_inp.to(f'cuda:{self.device}')
                    if self.use_half:
                        inp = inp.half()
                        encoder_word_pos_inp = encoder_word_pos_inp.half()
                        gsrm_word_pos_inp = gsrm_word_pos_inp.half()
                        gsrm_slf_attn_bias1_inp = gsrm_slf_attn_bias1_inp.half()
                        gsrm_slf_attn_bias2_inp = gsrm_slf_attn_bias2_inp.half()

                    backbone_out = self.net.backbone(inp)
                    prob_out = self.net.head(backbone_out, [encoder_word_pos_inp, gsrm_word_pos_inp, gsrm_slf_attn_bias1_inp, gsrm_slf_attn_bias2_inp])
                preds = {"predict": prob_out["predict"]}
            elif self.rec_algorithm == "SAR":
                valid_ratios = np.concatenate(valid_ratios)
                with torch.no_grad():
                    inp = torch.from_numpy(norm_img_batch)
                    valid_ratios_tensor = torch.from_numpy(valid_ratios)
                    if self.use_gpu:
                        inp = inp.to(f'cuda:{self.device}')
                        valid_ratios_tensor = valid_ratios_tensor.to(f'cuda:{self.device}')
                    if self.use_half:
                        inp = inp.half()
                    prob_out = self.net(inp, valid_ratios_tensor)

                    preds = self.net(inp, valid_ratios_tensor)
            elif self.rec_algorithm == "CAN":
                norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
                word_label_list = np.concatenate(word_label_list)
                inputs = [norm_img_batch, norm_img_mask_batch, word_label_list]
                with torch.no_grad():
                    inp = [torch.from_numpy(e) for e in inputs]
                    if self.use_gpu:
                        inp = [e.to(f'cuda:{self.device}') for e in inp]
                    if self.use_half:
                        inp = [e.half() for e in inp]
                    outputs = self.net(inp)
                    preds = [v.cpu().numpy() for v in outputs]
            else:
                with torch.no_grad():
                    inp = torch.from_numpy(norm_img_batch)
                    if self.use_gpu:
                        inp = inp.to(f'cuda:{self.device}')
                    if self.use_half:
                        inp = inp.half()
                    prob_out = self.net(inp)
                preds = prob_out.cpu().numpy() if not isinstance(prob_out, list) else [v.cpu().numpy() for v in prob_out]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res

def clean_license_plate(ocr_result):
    return ''.join(c for c in ocr_result if c.isalnum())

def warmup(text_recognizer, image_file_list):
    img_list = []
    for image_file in image_file_list[:text_recognizer.rec_batch_num * 5]:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        img_list.append(img)
    text_recognizer(img_list)

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    warmup(text_recognizer, image_file_list)
    valid_image_file_list = []
    img_list = []

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print(f"Error in loading image: {image_file}")
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)

    try:
        start_time = time.time()
        rec_res = text_recognizer(img_list)
        total_time = time.time() - start_time
    except Exception as e:
        print(e)
        sys.exit(1)

    results = []
    for ino in range(len(img_list)):
        print(f"Predicts of {valid_image_file_list[ino]}: {rec_res[ino]}")
        file_name = os.path.basename(valid_image_file_list[ino])
        text, conf = rec_res[ino]
        text = clean_license_plate(text)
        results.append({
            "file_name": file_name,
            "text": text,
            "conf": float(conf)
        })

    avg_time = 1000 * total_time / len(img_list)
    print(f"Predict time of {len(img_list)} images: {total_time:.3f}s, avg time: {avg_time:.2f}ms")

    output_json_path = args.output_json_path if hasattr(args, 'output_json_path') else './output/results.json'
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_json_path}")

if __name__ == '__main__':
    args = config
    main(args)
