 #!/bin/bash

# build all proprietary  networks
python3 -m trt_exporter -a
python3 -m dtrt_exporter

# build yolov8
mkdir -p /exports/yolov8
python3 -m export_yolov8 -w /model_repository/yolov8/yolov8s-27cls_1_1.pt -o /exports/yolov8/


## build yolov5 - still needed for testing
mkdir -p /exports/yolov5
python3 -m export_yolov5 -w /model_repository/yolov5/yolov5m-18cls.pt -o /exports/yolov5/

# build stgcn
mkdir -p /exports/stgcn
polygraphy convert /model_repository/stgcn/stgcn_w25_1_0.onnx  -o /exports/stgcn/stgcn_w25_1_0_fp16_b-1.engine --fp16\
  --trt-min-shapes input:[1,1,1,25,15,3] --trt-opt-shapes input:[1,1,1,25,15,3] --trt-max-shapes input:[8,1,1,25,15,3]

# build magface
mkdir -p /exports/magface
polygraphy convert /model_repository/magface/magface_iresnet18_norm.onnx -o /exports/magface/magface_iresnet18_norm_b-1_fp16.engine --fp16 \
 --trt-min-shapes input:[1,3,112,112] --trt-opt-shapes input:[2,3,112,112] --trt-max-shapes input:[8,3,112,112] --tensor-datatypes input:float16 output:float16 --precision-constraints obey

# build ediffiqa
mkdir -p /exports/ediffiqa
polygraphy convert /model_repository/ediffiqa/ediffiqa_iresnet18_norm.onnx -o /exports/ediffiqa/ediffiqa_iresnet18_norm_b-1_fp16.engine --fp16 \
 --trt-min-shapes input:[1,3,112,112] --trt-opt-shapes input:[2,3,112,112] --trt-max-shapes input:[8,3,112,112] --tensor-datatypes input:float16 output:float16 --precision-constraints obey

## build yolo exper model - needs to be optimized later
mkdir -p /exports/expert-detect
# convert to onnx with dynamic axis
yolo export model=/model_repository/expert-detect/yolov8m-guns-1280_1_1.pt format=onnx device=cpu dynamic=True  imgsz=[704,1280]
polygraphy convert /model_repository/expert-detect/yolov8m-guns-1280_1_1.onnx -o /exports/expert-detect/yolov8m-guns-1280_1_1_fp16_b-1_rtx3050.engine --fp16 \
 --trt-min-shapes images:[1,3,704,1280] --trt-opt-shapes images:[2,3,704,1280] --trt-max-shapes images:[8,3,704,1280] --tensor-datatypes images:float16 output0:float16  --precision-constraints obey

## build onnx networks
#mkdir -p /exports/yunet
#polygraphy  convert -o /exports/yunet/yunet.engine  /model_repository/yunet/yunet.onnx
#
#mkdir -p /exports/sface
#polygraphy  convert -o /exports/sface/sface.engine  /model_repository/sface/sface.onnx
#
## build yolov5
#mkdir -p /exports/yolov5
#python3 -m export_yolov5 -w /model_repository/yolov5/yolov5m-18cls.pt -o /exports/yolov5/
#
#
#
#
## build stgcn
#mkdir -p /exports/stgcn
#polygraphy convert /model_repository/stgcn/stgcn_32.onnx  -o /exports/stgcn/stgcn_fp32_b-1.engine \
# --trt-min-shapes input:[1,1,1,40,17,3] --trt-opt-shapes input:[1,1,1,40,17,3] --trt-max-shapes input:[8,1,1,40,17,3]
#
#mkdir -p /exports/rtmpose
#polygraphy convert /model_repository/rtmpose/rtmpose_32.onnx  -o /exports/rtmpose/rtmpose_fp16_b-1.engine --trt-min-shapes input:[1,3,256,192] --trt-opt-shapes input:[20,3,256,192] --trt-max-shapes input:[40,3,256,192] --fp16
