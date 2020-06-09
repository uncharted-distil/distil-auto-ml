FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.5.18

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
#RUN pip3 install -e git+https://github.com/cdbethune/D3M-Unsupervised.git@f6d036b6f7fcedcf809fde0ec744f1862308fced#egg=D3MUnsupervised
#RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@82698e594a9b4b4cfae86bcab9a98ffc47c3e131#egg=DistilPrimitives
#RUN pip3 install -e git+https://github.com/uncharted-distil/distil-mi-ranking.git@2ce1d22e1b0c212af043a3fcd48079d8000647db#egg=DistilMIRanking
#RUN pip3 install -e git+https://github.com/NewKnowledge/duke-d3m-wrapper.git@1c5e5f20fced72f042581a2f9cfca41557a2d85d#egg=DukeD3MWrapper
#RUN pip3 install -e git+https://github.com/NewKnowledge/simon-d3m-wrapper.git@dd573f75402d5246882c1f9fa2809300a74a1eb8#egg=SimonD3MWrapper
#RUN pip3 install -e git+https://github.com/NewKnowledge/goat-d3m-wrapper.git@f24adb14d6ca228d54f1d0adb5309e0704b274d4#egg=GoatD3MWrapper
COPY . .

ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python3", "/app/main.py"]
