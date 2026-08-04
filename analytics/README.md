# LumixAI - Analytics Application Setup

# docker setup
# jetson
sudo curl -L "https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-aarch64" -o /usr/local/bin/docker-compose
# cpu
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart 

# Enable jeston api for compilation on non root
sudo chown -R $USER /usr/src/jetson_multimedia_api/
sudo chgrp -R $USER /usr/src/jetson_multimedia_api/
	
### Clone the app 
```
cd ~ 
git clone git@github.com:lumixai/lumixai_analytics.git
cd lumixai_analytics
```
### App volumes - create the following directories    
```
mkdir ~/dev ~/configs
ln -s /dev/shm $HOME/dev/shm
```
### Create library for images/videos
- Put your images/movies under $HOME/analytics/data/
- By default it look for videos. if you want to run on mp4 run compose by overiding VIDEO_PATH: `VIDEO_TEST='False'`

```
mkdir $HOME/analytics/
mkdir $HOME/analytics/data/
```
- Results are under: `$HOME/analytics/results/`

### Now you are ready to setup your model 
```
./setup.sh -a -z
#./setup.sh -a <Analytics mode: cuda/cpu/jetson/orin>
# cuda(a): platforms with CUDA support 
# cpu(c): non jetson platform 
# jetson(j): jetson platform 
# orin(o): orin platform

#-z is requiered only on first build or when c files were modified
```

### Generate TensorRT engine
Notes:
- You need to install the required packages (detection/yolov5/requirements.txt). 
- For each batch size you will need to export different engine.
- FP32 is much slower than PyTorch, so mainly we want to use FP16 (call with --half flag).
- For faster export increase the workspace size.
- If you are using Xavier nx, keep the default workspace size (since we don't have enough memory).

Arguments:
```
cd analyzer_manager/app/detection/yolov5
python3 export_tensorrt.py -h
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     model.pt path
  --imgsz IMGSZ [IMGSZ ...]
                        image (h, w)
  --batch-size BATCH_SIZE
                        batch size
  --verbose             TensorRT: verbose log
  --half                fp16 engine
  --workspace WORKSPACE
                        TensorRT workspace size (GB)

```

Example for exporting yolov5m FP16 engine with batch size of 12:
```
python3 export_tensorrt.py --weights yolov5m.pt --half --batch-size 12
```
For this example the engine will be exported as yolov5m_fp16_b12.engine

Using the exported engine:
To use the engine, you only need to change the path in the yolov5 configuration arguments (the weights parameter):
```
weights = WORKDIR + '/detection/yolov5/yolov5m_fp16_b12.engine' 
```

also in parseYolov5:
```
if "weights"    in msg: yolo_args.weights    = WORKDIR + '/detection/yolov5/'+ msg["weights"]+'_fp16_b12.engine'
```

### Build your model
```
docker-compose build
```
### Remove your model 
```
docker-compose down
```
### Run your model 
```
docker-compose up
```

## Testing
The testing environment is using pytest to run all the tests inside [analyzer_manger/tests](analyzer_manger/tests).

To test the code in jetson, simple build and run the Docker for testing:
```
cd analyzer_manager

docker build -f ./Dockerfile.automation -t analyzer_tests .

docker run --runtime nvidia -it  analyzer_tests
```
