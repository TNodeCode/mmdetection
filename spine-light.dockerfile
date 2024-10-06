# Base image
FROM python:3.10

# Working directory
WORKDIR /app

# Get root priviledges
USER root

# Install necessary libraries
RUN apt-get update && apt-get install -y build-essential cmake git ffmpeg libsm6 libxext6

# Install dependencies
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
RUN python -m pip install -U openmim
RUN mim install mmengine mmcv==2.1

# Copy source code into image
COPY ./ ./

# Install mmdetection
RUN python -m pip install -v -e .
RUN python -m pip install -v -e . -r requirements/tracking.txt
RUN python -m pip install globox lap git+https://github.com/tnodecode/spineui#egg=trackeval&subdirectory=repositories/trackeval

# Command to run the app
CMD uvicorn api:app --host 0.0.0.0 --port 80