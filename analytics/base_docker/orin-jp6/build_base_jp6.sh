#!/bin/bash
echo "building base docker and uploading to docker hub"
docker build -f DockerfileMlBase.orin-jp6 -t  lumixai/analytic_base:0.1.0-orin-jp6 . && docker push lumixai/analytic_base:0.1.0-orin-jp6
