FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200630-050709

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

# mongodb is dumb, but is required for hyperparam tuning also base image is bionic not xenial (does this really matter?)
RUN wget -qO - https://www.mongodb.org/static/pgp/server-3.6.asc | sudo apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.6 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.6.list
RUN apt-get clean && apt-get update
RUN apt-get install -y mongodb-org
RUN mkdir ./sherpa_temp

# lzo support since it is not currently built into the d3m image
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y liblzo2-dev

ARG CACHEBUSTER=0
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-fuzzy-join.git@d171c9dc29d699dba10c1fdd5f00db8bbdd37f7d#egg=DistilFuzzyJoin

# Update as needed when new versions not built into base image
RUN rm -rf /src/distil-primitives
RUN rm -rf /app/src/distilprimitives
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@cbc2761fd4cc0e67b14dddebeaed2d9b873c141c#egg=DistilPrimitives
RUN pip3 install -e git+https://github.com/kungfuai/d3m-primitives.git@66591120ff04cd62d4b52239b9ba9fd3c9a91b7e#egg=kf-d3m-primitives

# manually install python-lzo, since it is installed via d3m build process and can't be included in the setup.py
RUN pip3 install python-lzo==1.12

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
