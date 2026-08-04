import torch
from predict_rec_model_batch_modular import full_ocr_model, get_image_file_list, ModelBuilder, Preprocessor, InferenceEngine, Postprocessor
import argparse
import cv2
import numpy as np

config = argparse.Namespace(
    limited_max_width=1280,
    limited_min_width=16,
    max_text_length=10,
    output_json_path='/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/finetuned-v4-owl_crops-torch.json',
    rec_algorithm='CRNN',
    rec_batch_num=1,
    rec_char_dict_path='./ppocr/utils/en_dict.txt',
    rec_char_type='en',
    rec_image_inverse=True,
    rec_image_shape='3,48,320',
    rec_model_path='./en_ptocr_v4_rec_infer.pth',
    rec_yaml_path='./configs/rec/PP-OCRv4/en_PP-OCRv4_rec_LPR.yml',
    scales=[8, 16, 32],
    show_log=False,
    use_space_char=True,
    use_gpu=True,
    device=1,
    use_half=True,
)

image_dir = '/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/lp/'
model_path = f'/home/noamr/analytics_research/our_ocr/ocr_fp16.onnx'


image_file_list = get_image_file_list(image_dir)
image_file_list.sort()
image_file_list = image_file_list[:config.rec_batch_num]
print(image_file_list)
valid_image_file_list = []
img_list = []
for image_file in image_file_list:
    img = cv2.imread(image_file)
    img_list.append(img)


model_builder = ModelBuilder(config)
preprocessor = Preprocessor(config)
inference_engine = InferenceEngine(model_builder.net, config)
postprocessor = Postprocessor(config)


preprocessed_data = preprocessor.preprocess(img_list)

print(f'preprocessed_data shape: {preprocessed_data.shape}')

# save as numpy
np.save('/home/noamr/analytics_research/our_ocr/preprocessed_input_data.npy', preprocessed_data.cpu().numpy())

torch.onnx.export(inference_engine,               # model being run
                preprocessed_data,         # model input (or a tuple for multiple inputs)
                model_path,        # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=12,    # the ONNX version to export the model to
                do_constant_folding=False,  # whether to execute constant folding for optimization
                input_names=['input'],   # the model's input names
                output_names=['output'],  # the model's output names
                dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                              'output': {0: 'batch_size'}},
                training=torch.onnx.TrainingMode.EVAL)

