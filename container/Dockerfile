# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install all of the packages
#RUN wget https://bootstrap.pypa.io/3.3/get-pip.py && python3.6 get-pip.py
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install flask
RUN pip install gevent
RUN pip install gunicorn
RUN pip install tensorflow
RUN pip install keras
RUN pip install gensim
RUN pip install tqdm
RUN pip install nltk
RUN pip install emoji
RUN pip install sagemaker

RUN (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .)
RUN rm -rf /root/.cache
        
# Env Variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY nn /opt/program
WORKDIR /opt/program
