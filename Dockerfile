FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200612-000030

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt
# mongodb is dumb, but is required for hyperparam tuning
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.2.list
RUN apt-get update
RUN apt-get install -y mongodb-org
RUN mkdir ./sherpa_temp


ARG CACHEBUSTER=0
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-fuzzy-join.git@d171c9dc29d699dba10c1fdd5f00db8bbdd37f7d#egg=DistilFuzzyJoin

# Update as needed when new versions not built into base image
#RUN pip3 uninstall -y ShapExplainers
#RUN rm -rf /src/distil-primitives
#RUN rm -rf /app/src/distilprimitives
#RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@82698e594a9b4b4cfae86bcab9a98ffc47c3e131#egg=DistilPrimitives
#RUN pip3 install -e git+https://github.com/kungfuai/d3m-primitives.git@67e436e383c59dccb63bf683edb6f6a122b5c1e7#egg=kf-d3m-primitives

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
