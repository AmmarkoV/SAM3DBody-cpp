#!/usr/bin/env bash
# This script builds and runs a docker image for local use.

#Although I dislike the use of docker for a myriad of reasons, due needing it to deploy on a particular machine
#I am adding a docker container builder for the repository to automate the process


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..
REPOSITORY=`pwd`

cd "$DIR"

NAME="sam3dbodycpp"
dockerfile_pth="$DIR"
# Mounted at /home/user/workspace.  Default: this repository.  MOUNT_PTH=/some/parent ./build_and_deploy.sh
# mounts a parent folder instead (e.g. one holding datasets/SAM3DBody-cpp, to keep existing absolute paths).
mount_pth="${MOUNT_PTH:-$REPOSITORY}"

export DOCKER_BUILDKIT=1


# Build context is the project root so the Dockerfile can COPY requirements.txt
# BuildKit's --ssh forwarding needs a live agent socket; start one if the shell
# has none (e.g. a plain SSH login where SSH_AUTH_SOCK isn't exported).
[ -n "$SSH_AUTH_SOCK" ] || eval "$(ssh-agent -s)"

# update tensorflow image
docker pull tensorflow/tensorflow:latest-gpu

# build and run tensorflow
docker build \
    --ssh default \
	-t $NAME \
	$dockerfile_pth \
	--build-arg user_id=$UID

# --cpus 32 \
#--mount type=tmpfs,destination=/home/user/ram,tmpfs-mode=1777,size=140G,mpol=bind,huge=always \
# was --mount type=tmpfs,destination=/home/user/ram,tmpfs-mode=1777 \
#--tmpfs /home/user/ram:rw,size=140g,mode=1777 \
# The renderer opens an X display through GLX, so for GPU OpenGL the container uses the host's X
# server (which must run on the NVIDIA driver); allow local clients first.
xhost +local: > /dev/null 2>&1 || echo "xhost failed: no host X server? OpenGL will not work in the container"
docker run -d \
	--gpus all \
	-e DISPLAY=${DISPLAY:-:0} \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
	--shm-size 32G \
    --cap-add=SYS_NICE \
    --mount type=tmpfs,destination=/home/user/ram,tmpfs-mode=1777 \
	-it \
	--name $NAME-container \
	-v $mount_pth:/home/user/workspace \
    -v /storage:/storage \
    -p 6065:6065 \
	$NAME


docker ps -a

OUR_DOCKER_ID=`docker ps -a | grep $NAME | cut -f1 -d' '`
echo "Our docker ID is : $OUR_DOCKER_ID"

echo "To monitor resource-consumption use: docker stats $NAME-container"
echo "Attaching it using : docker attach $OUR_DOCKER_ID"
docker attach $OUR_DOCKER_ID



exit 0
