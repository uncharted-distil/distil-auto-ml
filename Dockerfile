FROM "ubuntu:bionic"

ARG BRANCH_NAME=__UNSET__
ENV BRANCH_NAME=${BRANCH_NAME}

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update -qq \
    && apt-get install -y -qq \
    build-essential \
    git \
    python-dev \
    python-pip \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libcurl4-openssl-dev \
    libssl-dev \
    tzdata \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libncurses5-dev \
    swig \
    && apt-get autoremove -y \
    && apt-get purge -y \
    && apt-get clean -y

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

# TODO: not this
RUN pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/d3m.git@v2019.4.4

# Common primitives
RUN pip3 install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/common-primitives.git@v0.3.0

# Our primitives
# cython there because no good way to spec build time dependency
RUN pip3 install --upgrade setuptools pip cython==0.29.3
RUN echo HAH
RUN pip3 install git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives

RUN pip3 install git+https://gitlab.com/datadrivendiscovery/d3m.git@v2019.4.4

# TODO: fix in build
RUN apt-get -qq update -qq \
    && apt-get install -y -qq build-essential libcap-dev
RUN pip3 install python-prctl

# Put everything in
COPY .git /.git
COPY . .

#ENV D3M_BASE_IMAGE_NAME=_none_
#ENV D3M_BASE_IMAGE_DIGEST=_none_
#ENV D3M_IMAGE_NAME=_none_
#ENV D3M_IMAGE_DIGEST=_none_

RUN pip3 install -e /app
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
