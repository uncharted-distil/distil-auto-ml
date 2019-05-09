FROM registry.gitlab.com/datadrivendiscovery/images/base:ubuntu-bionic-python36

ARG BRANCH_NAME=__UNSET__
ENV BRANCH_NAME=${BRANCH_NAME}

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
    pip3 install resampy soundfile && \
    apt-get install -y curl && \
    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt && \
    mv vggish_model.ckpt /app/third_party/audioset/vggish_model.ckpt

# TODO: not this
RUN pip3 install git+https://gitlab.com/datadrivendiscovery/d3m.git@v2019.4.4

# Common primitives
RUN pip3 install git+https://gitlab.com/datadrivendiscovery/common-primitives.git@11f24bf517a98a57c9175d71c73db8ed0be69167

# TODO: fix in build
RUN apt-get -qq update -qq \
    && apt-get install -y -qq build-essential libcap-dev
RUN pip3 install python-prctl

RUN pip3 install cython==0.29.3
RUN echo '          '
# Put everything in
COPY .git /.git
COPY . .

RUN echo '                       '
# Our primitives
RUN pip3 install git+https://github.com/uncharted-distil/distil-primitives.git@remove_mmap2

RUN pip3 install -e /app
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
