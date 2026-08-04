import numpy as np
from PIL import Image


def transform_args(attributes_args):
    model_kwargs = dict()
    model_kwargs["num_att"] = attributes_args.num_att
    model_kwargs["last_conv_stride"] = attributes_args.last_conv_stride
    model_kwargs["im_size"] = attributes_args.im_size
    model_kwargs["use_calibration"] = attributes_args.use_calibration
    return model_kwargs


def pad_image_sqr_pil(input_image):
    rect = max([input_image.size[0], input_image.size[1]])
    nh, nw = rect, rect

    w, h = input_image.size
    fh, fw = nh / h, nw / w
    pad_w_l, pad_h_u = 0, 0
    image_padded = Image.new(input_image.mode, (nw, nh), (0, 0, 0))
    if fh < fw:
        tmp_h = h
        tmp_w = int(h * (nw / nh))
        pad_w_l = int((tmp_w - w) / 2)
        image_padded.paste(input_image, (pad_w_l, 0))
    else:
        tmp_h = int(w * (nh / nw))
        tmp_w = w
        pad_h_u = int((tmp_h - h) / 2)
        image_padded.paste(input_image, (0, pad_h_u))

    return image_padded


def pad_image_sqr_tensor(input_image):
    import torch.nn.functional as f

    _, h, w = input_image.shape
    rect = max([h, w])
    nh, nw = rect, rect

    fh, fw = nh / h, nw / w
    pad_w_l, pad_h_u = 0, 0
    #   pad (l, r, t, b)
    if fh < fw:
        tmp_h = h
        tmp_w = int(h * (nw / nh))
        pad_w_l = int((tmp_w - w) / 2)
        pad_w_r = nw - pad_w_l - w
        image_padded = f.pad(input_image, (pad_w_l, pad_w_r, 0, 0))
    elif fh > fw:
        tmp_h = int(w * (nh / nw))
        tmp_w = w
        pad_h_u = int((tmp_h - h) / 2)
        pad_h_d = nh - pad_h_u - h
        image_padded = f.pad(input_image, (0, 0, pad_h_u, pad_h_d))
    else:
        image_padded = input_image
    return image_padded


def identity(input_image):
    return input_image


def fetch_transforms(attributes_args, to_tensor=False):
    import torchvision.transforms as transforms

    if attributes_args.no_sqr_pad:
        pad_func = identity
    else:
        pad_func = pad_image_sqr_tensor
    #normalize = transforms.Normalize(mean=attributes_args.mean, std=attributes_args.std)

    if to_tensor:
        transforms_out = transforms.Compose(
            [
                transforms.ToTensor(),
                # normalize,
                pad_func,
                transforms.Resize(attributes_args.im_size),
            ]
        )
    else:
        transforms_out = transforms.Compose(
            [
                # normalize,
                pad_func,
                transforms.Resize(attributes_args.im_size),
            ]
        )

    return transforms_out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
