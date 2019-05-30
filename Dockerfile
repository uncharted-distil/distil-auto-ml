FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.5.8-20190530-050006

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# For torch
ENV TORCH_MODEL_ZOO=/app

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

# Out-of-band Distil requirements
RUN apt update && \
    apt-get install -y ffmpeg build-essential libcap-dev curl && \
    pip3 install --upgrade pip cython==0.29.3 python-prctl==1.7 && \
    mkdir -p /app/third_party && \
    cd /app/third_party && \
    git clone https://github.com/tensorflow/models.git && \
    cd models && git checkout aecf5d0256806d4cb3b32fa87406d891e11dbe94 && cd .. && \
    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt && \
    mv vggish_model.ckpt /app/third_party/models/research/audioset/vggish_model.ckpt && \
    mv /app/third_party/models/research/audioset /app/third_party/audioset && \
   rm -rf /app/third_party/models

# apt cleanup
RUN rm -rf /var/lib/apt/lists/* /opt/* /tmp/*

# Our primitives
RUN pip3 install git+https://github.com/uncharted-distil/distil-primitives.git@collab-filter-cleanup#egg=DistilPrimitives

# Put everything in
COPY .git /.git
COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
