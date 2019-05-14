FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4-20190509-004421

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# For torch
ENV TORCH_MODEL_ZOO=/app

# timezone-related fixes
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
   dpkg-reconfigure --frontend noninteractive tzdata

# apt cleanup
RUN rm -rf /var/lib/apt/lists/* /opt/* /tmp/*

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

# Install exline requirements
COPY build.sh .
RUN sh build.sh

RUN apt update && \
    apt-get install -y ffmpeg && \
    pip3 install resampy==0.2.1 soundfile==0.10.2 && \
    apt-get install -y curl && \
    mkdir -p /app/third_party/audioset && \
    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt && \
    mv vggish_model.ckpt /app/third_party/audioset/vggish_model.ckpt

# TODO: fix in build
RUN apt-get -qq update -qq \
    && apt-get install -y -qq build-essential libcap-dev
RUN pip3 install python-prctl
RUN pip3 install --upgrade pip cython==0.29.3

# Put everything in
COPY .git /.git
COPY . .

# Our primitives
RUN pip3 install git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives

RUN pip3 install -e /app
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
