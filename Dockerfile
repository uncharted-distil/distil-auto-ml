FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

ARG CACHEBUSTER=0
#RUN rm -rf /src/distil-primitives
#RUN rm -rf /app/src/distilprimitives
#RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@6f126d40bc8cfd2866c9477e7a0b1778a111487c#egg=DistilPrimitives
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-fuzzy-join.git@d171c9dc29d699dba10c1fdd5f00db8bbdd37f7d#egg=DistilFuzzyJoin
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-mi-ranking.git@2ce1d22e1b0c212af043a3fcd48079d8000647db#egg=DistilMIRanking

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
