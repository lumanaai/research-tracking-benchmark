#!/bin/bash


LOCKFILE="/tmp/run_exporter.lock"
DOCKER_IMAGE="lumixai/pf_exporter"
TAG=$1
NUM_WORKERS=${2:-2}
LOCK_DIR="/tmp/exporter_locks"

if [ -f "$LOCKFILE" ]; then
    echo "$(date): Another instance of run_exporter.sh is running. Exiting."
    exit 1
fi

# Create the lock file
touch "$LOCKFILE"
cleanup() {
    rm -f "$LOCKFILE"
    # Stop all workers if script is killed
    for i in $(seq 1 $NUM_WORKERS); do
        docker stop "pf_exporter_$i" 2>/dev/null
    done
}
trap cleanup EXIT

# Ensure shared lock directory exists
mkdir -p "$LOCK_DIR"

# Log in to DockerHub
docker login -u readonlylumix -p dckr_pat_KChNg2_7FAcqFLTvGRkHlPQWQKE

# Pull the latest Docker image with the specified tag
docker pull "$DOCKER_IMAGE:$TAG"

# Launch N workers in parallel
echo "$(date): Launching $NUM_WORKERS workers with image $DOCKER_IMAGE:$TAG"
for i in $(seq 1 $NUM_WORKERS); do
    docker run --runtime nvidia \
        -v ~/.aws:/root/.aws \
        -v ~/logs:/logs \
        -v "$LOCK_DIR":/shared/locks \
        --rm --name "pf_exporter_$i" \
        "$DOCKER_IMAGE:$TAG" \
        python3 -u pf_exporter_s3.py &
done

# Wait for all workers to finish
wait
echo "$(date): All $NUM_WORKERS workers completed."
