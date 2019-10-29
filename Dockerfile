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

RUN pip3 install -e git+https://github.com/uncharted-distil/distil-mi-ranking.git@2ce1d22e1b0c212af043a3fcd48079d8000647db#egg=DistilMIRanking
RUN pip3 install -e git+https://github.com/NewKnowledge/pcafeatures-d3m-wrapper.git@4b31ed6098236ef7392768c45e4fa2f238124d3c#egg=PcafeaturesD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/datacleaning-d3m-wrapper.git@d2666de3de1406adfad4c74f4aacaf52ceccfced#egg=DatacleaningD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/duke-d3m-wrapper.git@1c5e5f20fced72f042581a2f9cfca41557a2d85d#egg=DukeD3MWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/unicorn-d3m-wrapper.git@865e2780e67c1f34ee2621748f557078d34eb15f#egg=UNICORNd3mWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/croc-d3m-wrapper.git@86695826d2481949c55e38020be222d231962806#egg=CROCd3mWrapper
RUN pip3 install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@2c95c371f4462ffae1cf8ea1ae6902ca23539adc#egg=SimonD3MWrapper

# Set common primitives version to use. Manually force uninstall since 'pip uninstall' fails - https://github.com/pypa/pip/issues/4438
RUN rm -rf /src/common-primitives
RUN pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@8f22abd534abcafc62bc585344c385c50ca5107f#egg=common-primitives

ARG CACHEBUSTER=0
RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives

COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]