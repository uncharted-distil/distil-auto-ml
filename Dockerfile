FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18-20200612-000030

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt


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
