FROM python:3.7

USER root
ARG RELEASE=master
RUN apt-get update && apt-get install -y apt-utils git vim

RUN git clone --branch $RELEASE --single-branch https://gitlab.com/msc-business-analytics/filmoid.git

ENV HOME=/filmoid/app
WORKDIR ${HOME}

RUN pip3 install -r requirements.txt

COPY entrypoint.sh ${HOME}
COPY env_prep.py ${HOME}

CMD bash entrypoint.sh && tail -f /dev/null
#CMD tail -f /dev/null