FROM python:3.7

USER root
ARG RELEASE=master
RUN apt-get update && apt-get install -y apt-utils git vim

RUN git clone --branch $RELEASE --single-branch https://gitlab.com/msc-business-analytics/filmoid.git

ENV HOME=/filmoid
WORKDIR ${HOME}

RUN rm -rf ${HOME}/container ${HOME}/.git ${HOME}/docs
RUN rm ${HOME}/.gitignore ${HOME}/.gitlab-ci.yml  ${HOME}/ReadMe.md

RUN pip3 install -r requirements.txt

COPY entrypoint.sh ${HOME}
COPY env_prep.py ${HOME}

#CMD entrypoint.sh && tail -f /dev/null
CMD tail -f /dev/null