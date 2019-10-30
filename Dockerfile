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

# Set common primitives version to use. Manually force uninstall since 'pip uninstall' fails - https://github.com/pypa/pip/issues/4438
RUN rm -rf /src/common-primitives
RUN pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@8f22abd534abcafc62bc585344c385c50ca5107f#egg=common-primitives

# Seems like the dependency in distil primitives gets completely ignored if not explicitly installed
RUN pip3 install -e git+https://github.com/cdbethune/sklearn-pandas.git@c009c3a5a26f883f759cf123c0f5a509b1df013b#egg=sklearn-pandas

# Our primitives - not needed during eval since we use the frozen d3m primitive image
ARG CACHEBUSTER=0
RUN rm -rf /src/distil-primitives
RUN rm -rf /app/src/distilprimitives
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@6f126d40bc8cfd2866c9477e7a0b1778a111487c#egg=DistilPrimitives
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-fuzzy-join.git@d171c9dc29d699dba10c1fdd5f00db8bbdd37f7d#egg=DistilFuzzyJoin
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-mi-ranking.git@2ce1d22e1b0c212af043a3fcd48079d8000647db#egg=DistilMIRanking
RUN pip3 install -e git+https://github.com/NewKnowledge/pcafeatures-d3m-wrapper.git@bc34b607a15bdbb765a2f959a1b81bc5f23cd469#egg=PcafeaturesD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/datacleaning-d3m-wrapper.git@ff4d6213bb75e41807ff99bfab2bbcef45edb5d4#egg=DatacleaningD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/duke-d3m-wrapper.git@1c5e5f20fced72f042581a2f9cfca41557a2d85d#egg=DukeD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git@865e2780e67c1f34ee2621748f557078d34eb15f#egg=UNICORNd3mWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/croc-d3m-wrapper.git@86695826d2481949c55e38020be222d231962806#egg=CROCd3mWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@60088ab82217767cd77f8a540ba56a24cd85401b#egg=SimonD3MWrapper

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
