# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

FROM tensorflow/tensorflow:1.15.0-gpu-py3

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./beautify_pipeline/to_edit_files/imagenet_utils.py /usr/local/lib/python3.6/dist-packages/keras/applications/imagenet_utils.py

EXPOSE ${PORT}
CMD python service/app.py