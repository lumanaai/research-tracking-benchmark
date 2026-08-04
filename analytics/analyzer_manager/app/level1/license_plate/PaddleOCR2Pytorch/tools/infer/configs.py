import argparse

config = argparse.Namespace(
    image_dir='/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/lp/',
    limited_max_width=1280,
    limited_min_width=16,
    max_text_length=10,
    output_json_path='/media/4TBSSD/data/vehicle_id/license_plate_dataset/test/finetuned-v4-owl_crops-torch.json',
    rec_algorithm='CRNN',
    rec_batch_num=6,
    rec_char_dict_path='./ppocr/utils/en_dict.txt',
    rec_char_type='en',
    rec_image_inverse=True,
    rec_image_shape='3,48,320',
    rec_model_path='./en_ptocr_v4_rec_infer.pth',
    rec_yaml_path='./configs/rec/PP-OCRv4/en_PP-OCRv4_rec_LPR.yml',
    scales=[8, 16, 32],
    show_log=False,
    use_gpu=True,
    device='1',
    use_space_char=True,
)
