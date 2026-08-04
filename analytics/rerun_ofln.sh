#!/bin/bash
# build dockers
python3 setup_dockers.py -b -sd -t

# clean previous containers
docker rm -f inference_server offline_analytics

# run inference server
docker run --rm --runtime=nvidia --name inference_server --network host -v ${HOME}/assets/:/assets/ -d lumixai/inference_server
sleep 5

# run offline analytics
docker run --rm --name offline_analytics --network=host --runtime=nvidia -v /dev/shm:/dev/shm -v /tmp:/tmp -v ${HOME}/edge/analytics/analyzer_manager/assets/:/usr/src/app/configs/  lumixai/analytic_manager sh offline_main.sh
