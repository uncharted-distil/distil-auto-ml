FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225

ENV PYTHONPATH=$PYTHONPATH:/app
ENV DEBIAN_FRONTEND=noninteractive

# Install server requirements
WORKDIR /app
COPY server-requirements.txt .
COPY api api
RUN pip3 install -r server-requirements.txt

RUN apt-get update

# external dependencies for NK CROC primitive
RUN apt-get install -y libleptonica-dev libtesseract-dev tesseract-ocr tesseract-ocr-eng

# Our primitives - not needed during eval since we use the frozen d3m primitive image
ARG CACHEBUSTER=0
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-timeseries-loader.git@0d52a475f52848707a2b6d69fb3bae58409a784d#egg=DistilTimeSeriesLoader
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-mi-ranking.git@3960299b26d52f8859917eec0cb4d8f1eaaa1c3f#egg=DistilMIRanking
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-fuzzy-join.git@d171c9dc29d699dba10c1fdd5f00db8bbdd37f7d#egg=DistilFuzzyJoin
RUN pip3 install -e git+https://github.com/NewKnowledge/pcafeatures-d3m-wrapper.git@4b31ed6098236ef7392768c45e4fa2f238124d3c#egg=PcafeaturesD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/datacleaning-d3m-wrapper.git@d2666de3de1406adfad4c74f4aacaf52ceccfced#egg=DatacleaningD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/duke-d3m-wrapper.git@1c5e5f20fced72f042581a2f9cfca41557a2d85d#egg=DukeD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git@865e2780e67c1f34ee2621748f557078d34eb15f#egg=UNICORNd3mWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/croc-d3m-wrapper.git@86695826d2481949c55e38020be222d231962806#egg=CROCd3mWrapper

# Set common primitives version to use. Manually force uninstall since 'pip uninstall' fails - https://github.com/pypa/pip/issues/4438
RUN rm -rf /src/common-primitives
RUN pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@8f22abd534abcafc62bc585344c385c50ca5107f#egg=common-primitives

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]