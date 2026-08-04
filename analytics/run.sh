#!/bin/bash
#--wait-ready --kill-timeout 3000 --listen-timeout 5000

#!/bin/bash
echo "use: run build / down / up args or setup/tests/exporter"

build=0
up=0
down=0
rebuild=0
tests=0
exporter=0
setup=0

while getopts "sbudrte" opt; do
  case "$opt" in
    b) build=1 ;;
    u) up=1 ;;
    d) down=1 ;; 
    r) rebuild=1 ;;
    t) tests=1 ;;
    e) exporter=1 ;;
    s) setup=1 ;;
  esac
done
shift $(( OPTIND - 1 ))

if [[ $rebuild == 1 ]]; then
    echo "Rebuild"
    build=1
    up=1
    down=1
fi

if [[ $setup == 1 ]]; then
    echo "setting up dockers"
    if [[ $build == 1 ]]; then
      python3 setup_dockers.py -b
    else
      python3 setup_dockers.py
    fi
fi


if [[ $down == 1 ]]; then     
    echo "down"
    docker-compose down     
fi 
if [[ $build == 1 ]]; then 
    echo "build"
    docker-compose build $@
fi
if [[ $up == 1 ]]; then     
    echo "up"
    docker-compose up 
fi
if [[ $tests == 1 ]]; then
    echo "run tests"
    pushd analyzer_manager
    docker build -f DockerfileAutomation -t tests .
    docker run --runtime nvidia -it tests python3 -m pytest ./tests/analytics/test_basic -m "not extended"
    popd
fi
if [[ exporter == 1 ]]; then
    echo "run tests"
    pushd analyzer_manager
    docker build -f DockerfileExporter -t exporter .
    docker run --runtime nvidia -it exporter
    popd
fi

