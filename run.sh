#!/usr/bin/env sh

NAME=aluan
if [ ! "$(docker ps -q -f name=$NAME)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$NAME)" ]; then
        docker rm $NAME
    fi
    docker run -d -p 8887:8888 --name $NAME \
      -v ~/Github/PythonProjects/alu/:/home/jovyan/alu \
      -v ~/Github/PythonProjects/alu/zipline:/home/jovyan/.zipline \
      qfeng/zipline start-notebook.sh --NotebookApp.token=''
fi
