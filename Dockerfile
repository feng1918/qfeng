FROM jupyter/scipy-notebook

LABEL maintainer="feng"

RUN pip install --upgrade pip
RUN pip --no-cache-dir install \
  lxml
RUN pip --no-cache-dir install \
  zipline \
  tushare
RUN mkdir .zipline

