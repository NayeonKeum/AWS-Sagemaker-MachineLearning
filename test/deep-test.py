#!/usr/bin/env python
# coding: utf-8

# # SageMaker/DeepAR demo on electricity dataset
#
# This notebook complements the [DeepAR introduction notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/deepar_synthetic/deepar_synthetic.ipynb).
#
# Here, we will consider a real use case and show how to use DeepAR on SageMaker for predicting energy consumption of 370 customers over time, based on a [dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) that was used in the academic papers [[1](https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips29/reviews/526.html)] and [[2](https://arxiv.org/abs/1704.04110)].
#
# In particular, we will see how to:
# * Prepare the dataset
# * Use the SageMaker Python SDK to train a DeepAR model and deploy it
# * Make requests to the deployed model to obtain forecasts interactively
# * Illustrate advanced features of DeepAR: missing values, additional time features, non-regular frequencies and category information
#
# Running this notebook takes around 40 min on a ml.c4.2xlarge for the training, and inference is done on a ml.m4.xlarge (the usage time will depend on how long you leave your served model running).
#
# For more information see the DeepAR [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) or [paper](https://arxiv.org/abs/1704.04110),

# In[69]:
from __future__ import print_function


import sys
from urllib.request import urlretrieve
import zipfile
from dateutil.parser import parse
import json
from random import shuffle
import random
import datetime
import os

import boto3
import s3fs
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import IntSlider, FloatSlider, Checkbox

from sagemaker import get_execution_role

import ntpath

# In[70]:


# set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


# In[71]:
boto3.setup_default_session(profile_name='default',region_name='us-east-2')

smclient = boto3.Session().client('sagemaker')


# boto3.Session().client('sagemaker')

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(),sagemaker_client=smclient)



# Before starting, we can override the default values for the following:
# - The S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.
# - The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these.

# In[72]:


s3_bucket = 'sagemaker-us-east-2-469432985395'  # replace with an existing bucket if needed
s3_prefix = 'smedu'    # prefix used for all data stored within the bucket



#role = get_execution_role() # IAM role to use by SageMaker

role="arn:aws:iam::469432985395:role/service-role/AmazonSageMaker-ExecutionRole-20201121T104674"
print(role)



# In[73]:


#region = sagemaker_session.boto_region_name
region = boto3.Session().region_name
print(region)

s3_data_path = "s3://{}/{}/data".format(s3_bucket, s3_prefix)
s3_output_path = "s3://{}/{}/output".format(s3_bucket, s3_prefix)


# Next, we configure the container image to be used for the region that we are running in.

# In[74]:


image_name = sagemaker.amazon.amazon_estimator.get_image_uri(region, "forecasting-deepar", "latest")


# ### Import electricity dataset and upload it to S3 to make it available for Sagemaker

# As a first step, we need to download the original data set of from the UCI data set repository.

freq = '1H'

local_data_dir_path = "../data/"
rid_targets = {}
def find_rid_targets(dirname):

    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                find_rid_targets(full_filename)
            else:

                filename = ntpath.basename(full_filename)
                rId = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[-1]

                if ext != '.csv':
                    continue

                files = rid_targets.get(rId)
                if files:
                    files.append(full_filename)
                else:
                    rid_targets[rId] = [full_filename]

    except PermissionError:
        pass

find_rid_targets(local_data_dir_path)

timeseries = []
i = 0

for n in rid_targets:
    j = 0
    for p in rid_targets[n]:

        data = pd.read_csv(p, index_col=0, parse_dates=['Timestamp'])
        data.columns = [n]

        data_kw = data.resample(freq).sum()

        if j == 0:
            timeseries.append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))
        else:
            timeseries[i] = timeseries[i].append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))
        j += 1
    i += 1


# Let us plot the resulting time series for the first ten customers for the time period spanning the first two weeks of 2014.

"""
fig, axs = plt.subplots(1, 2, figsize=(30, 20), sharex=True)
axx = axs.ravel()
for i in range(0, 1):
    timeseries[i].loc["2019-11-01":"2019-11-14"].plot(ax=axx[i])
    axx[i].set_xlabel("Timestamp")
    axx[i].set_ylabel("kW consumption")
"""

# ### Train and Test splits
#
# Often times one is interested in evaluating the model or tuning its hyperparameters by looking at error metrics on a hold-out test set. Here we split the available data into train and test sets for evaluating the trained model. For standard machine learning tasks such as classification and regression, one typically obtains this split by randomly separating examples into train and test sets. However, in forecasting it is important to do this train/test split based on time rather than by time series.
#
# In this example, we will reserve the last section of each of the time series for evalutation purpose and use only the first part as training data.

# we predict for 7 days
prediction_length = 7 * 24

# we also use 7 days as context length, this is the number of state updates accomplished before making predictions
context_length = 7 * 24


# We specify here the portion of the data that is used for training: the model sees data from 2014-01-01 to 2014-09-01 for training.


start_training_time = "2019-08-01 00:00:00"
now = datetime.datetime.now()
end_training_time = now.strftime('%Y-%m-%d 00:00:00')

start_dataset = pd.Timestamp(start_training_time, freq=freq)
end_training = pd.Timestamp(end_training_time, freq=freq)


# The DeepAR JSON input format represents each time series as a JSON object. In the simplest case each time series just consists of a start time stamp (``start``) and a list of values (``target``). For more complex cases, DeepAR also supports the fields ``dynamic_feat`` for time-series features and ``cat`` for categorical features, which we will use  later.


training_data = [
    {
        "start": str(start_dataset),
        "target": ts[start_dataset:end_training - 1*start_dataset.freq].tolist()  # We use -1, because pandas indexing includes the upper bound
    }
    for ts in timeseries
]
print(len(training_data))


# As test data, we will consider time series extending beyond the training range: these will be used for computing test scores, by using the trained model to forecast their trailing 7 days, and comparing predictions with actual values.
# To evaluate our model performance on more than one week, we generate test data that extends to 1, 2, 3, 4 weeks beyond the training range. This way we perform *rolling evaluation* of our model.

# In[82]:


num_test_windows = 4


end=(num_test_windows + 1*2)
test_data = [
    {
        "start": str(start_dataset),
        "target": ts[start_dataset:end_training + k * prediction_length * start_dataset.freq].tolist()
    }
    for k in range(1, end)
    for ts in timeseries
]
print(len(test_data))


def write_dicts_to_file(path, data):
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))


# In[85]:


write_dicts_to_file("train.json", training_data)
write_dicts_to_file("test.json", test_data)


s3 = boto3.resource('s3')
def copy_to_s3(local_file, s3_path, override=False):
    assert s3_path.startswith('s3://')
    split = s3_path.split('/')
    bucket = split[2]
    path = '/'.join(split[3:])
    buk = s3.Bucket(bucket)

    if len(list(buk.objects.filter(Prefix=path))) > 0:
        if not override:
            print('File s3://{}/{} already exists.\nSet override to upload anyway.\n'.format(s3_bucket, s3_path))
            return
        else:
            print('Overwriting existing file')
    with open(local_file, 'rb') as data:
        print('Uploading file to {}'.format(s3_path))
        buk.put_object(Key=path, Body=data)


# In[87]:


copy_to_s3("train.json", s3_data_path + "/train/train.json")
copy_to_s3("test.json", s3_data_path + "/test/test.json")
