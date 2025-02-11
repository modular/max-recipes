#!/bin/bash

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")

if [ "$NUM_GPUS" -gt 0 ] && nvidia-smi >/dev/null 2>&1; then
    export PROFILE="gpu"
    echo "Detected $NUM_GPUS GPU(s). Using GPU profile."
else
    export PROFILE="cpu"
    echo "No GPUs detected. Using CPU profile."
fi

case "$1" in
"app")
    echo "Starting the app on $PROFILE ..."
    docker compose --profile $PROFILE up
    ;;
"stop")
    echo "Stopping containers with $PROFILE ..."
    docker compose --profile $PROFILE down
    ;;
"clean")
    echo "Cleaning up containers with $PROFILE ..."
    docker compose --profile $PROFILE down -v

    if docker images -q modular/max-openai-api:${MAX_OPENAI_API_VERSION:-nightly} >/dev/null; then
        docker rmi -f $(docker images -q modular/max-openai-api:${MAX_OPENAI_API_VERSION:-nightly})
    fi
    if docker images -q max-serve-continuous-chat-ui:latest >/dev/null; then
        docker rmi -f $(docker images -q max-serve-continuous-chat-ui:latest)
    fi
    ;;
*)
    echo "Usage: $0 {app|stop|clean}"
    echo "  app   - Start the application"
    echo "  stop  - Stop the application"
    echo "  clean - Stop and remove containers"
    exit 1
    ;;
esac
