# Exporter docker

# introduction
the exporter docker is a tool for exporting weights to tensor-rt. 
it has no inputs, but should be set to search for internal /exports  and all the files in it, so the /export dir should be given as a mount
in the export folder, it searches for all .pt files, and parse them according to their prefix to understand the network to be exported.

if a json file exists with the name <network_name>_export_params.json, it uses it as input for the conversion process. 
see example yolov5_export_params.json

## Building the docker
use "DockerfileExporter" and build from analyzer_manager folder:
```
docker build -f DockerfileExporter -t analytics_exporter .
```

## Runing the docker
use the following options when running:
--gpus all : for accessing the GPU for the conversion
-v <exports_path>:/exports : for data folder
for example:

```
docker run --runtime nvidia -v ~/exports:/exports  analytics_exporter
```

### Yolo v5 parameters:

image_sz : size of input image ([h w]), default [384, 640]
batch_sz : number of images in batch, default 8
half     : whether to use fp16, default True
workspace: Size of workspace in GB, default 2
verbose  : verbosity, default False
weights  : ignored
