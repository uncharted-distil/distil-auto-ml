FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-stable-20201201-223410

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# install pre-requisite libs
# mongodb is dumb, but is required for hyperparam tuning also base image is bionic not xenial (does this really matter?)
RUN wget -qO - https://www.mongodb.org/static/pgp/server-3.6.asc | sudo apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.6 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.6.list
RUN apt-get clean && apt-get update
RUN apt-get install -y mongodb-org
# snappy for parquet compression
RUN apt-get install libsnappy-dev

RUN mkdir ./sherpa_temp

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

# pass the current date/time as a build arg to force a rebuild of everything after this line
ARG CACHEBUSTER=0

# Update as needed when new versions not built into base image
#RUN rm -rf /src/distil-primitives
#RUN rm -rf /app/src/distilprimitives
#RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@ae484c2718a3e73d94bfc4bbac71fafab9428174#egg=DistilPrimitives
# Pass the optional gpu version for mxnet - see the kf-d3m-primitives setup.py for valid values
#ARG GPU_MXNET=""
#RUN pip3 install -e git+https://github.com/kungfuai/d3m-primitives.git@e1fd17c01b7662653acdde83211a6df89ffcfaaf#egg=kf-d3m-primitives$GPU_MXNET

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
