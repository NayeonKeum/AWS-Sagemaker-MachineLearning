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

# Let's have a look to what we just wrote to S3.

# In[88]:


s3filesystem = s3fs.S3FileSystem()
with s3filesystem.open(s3_data_path + "/train/train.json", 'rb') as fp:
    print(fp.readline().decode("utf-8")[:100] + "...")


estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_name=image_name,
    role=role,
    train_instance_count=1,
    #train_use_spot_instances = True,
    train_instance_type='ml.c4.2xlarge',
    base_job_name='deepar-electricity-demo',
    output_path=s3_output_path
)

# Next we need to set the hyperparameters for the training job. For example frequency of the time series used, number of data points the model will look at in the past, number of predicted data points. The other hyperparameters concern the model to train (number of layers, number of cells per layer, likelihood function) and the training options (number of epochs, batch size, learning rate...). We use default parameters for every optional parameter in this case (you can always use [Sagemaker Automated Model Tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) to tune them).

# In[90]:


hyperparameters = {
    "time_freq": freq,
    "epochs": "400",
    "early_stopping_patience": "40",
    "mini_batch_size": "64",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length)
}

# In[91]:


estimator.set_hyperparameters(**hyperparameters)

# We are ready to launch the training job. SageMaker will start an EC2 instance, download the data from S3, start training the model and save the trained model.
#
# If you provide the `test` data channel as we do in this example, DeepAR will also calculate accuracy metrics for the trained model on this test. This is done by predicting the last `prediction_length` points of each time-series in the test set and comparing this to the actual value of the time-series.
#
# **Note:** the next cell may take a few minutes to complete, depending on data size, model complexity, training options.

# In[92]:


data_channels = {
    "train": "{}/train/".format(s3_data_path),
    "test": "{}/test/".format(s3_data_path)
}

estimator.fit(
    inputs=data_channels, wait=True)


# Since you pass a test set in this example, accuracy metrics for the forecast are computed and logged (see bottom of the log).
# You can find the definition of these metrics from [our documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html). You can use these to optimize the parameters and tune your model or use SageMaker's [Automated Model Tuning service](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) to tune the model for you.

# ### Create endpoint and predictor

# Now that we have a trained model, we can use it to perform predictions by deploying it to an endpoint.
#
# **Note: Remember to delete the endpoint after running this experiment. A cell at the very bottom of this notebook will do that: make sure you run it at the end.**

# To query the endpoint and perform predictions, we can define the following utility class: this allows making requests using `pandas.Series` objects rather than raw JSON strings.

# In[93]:


class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, content_type=sagemaker.content_types.CONTENT_TYPE_JSON, **kwargs)

    def predict(self, ts, cat=None, dynamic_feat=None,
                num_samples=100, return_samples=False, quantiles=["0.1", "0.5", "0.9"]):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.

        ts -- `pandas.Series` object, the time series to predict
        cat -- integer, the group associated to the time series (default: None)
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        return_samples -- boolean indicating whether to include samples in the response (default: False)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])

        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_time = ts.index[-1] + 1
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, ts.index.freq, prediction_time, return_samples)

    def __encode_request(self, ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
        instance = series_to_dict(ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None)

        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
            "quantiles": quantiles
        }

        http_request_data = {
            "instances": [instance],
            "configuration": configuration
        }

        return json.dumps(http_request_data).encode('utf-8')

    def __decode_response(self, response, freq, prediction_time, return_samples):
        # we only sent one time series so we only receive one in return
        # however, if possible one will pass multiple time series as predictions will then be faster
        predictions = json.loads(response.decode('utf-8'))['predictions'][0]
        prediction_length = len(next(iter(predictions['quantiles'].values())))
        prediction_index = pd.DatetimeIndex(start=prediction_time, freq=freq, periods=prediction_length)
        if return_samples:
            dict_of_samples = {'sample_' + str(i): s for i, s in enumerate(predictions['samples'])}
        else:
            dict_of_samples = {}
        return pd.DataFrame(data={**predictions['quantiles'], **dict_of_samples}, index=prediction_index)

    def set_frequency(self, freq):
        self.freq = freq


def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]


def series_to_dict(ts, cat=None, dynamic_feat=None):
    """Given a pandas.Series object, returns a dictionary encoding the time series.

    ts -- a pands.Series object with the target time series
    cat -- an integer indicating the time series category

    Return value: a dictionary
    """
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return obj


# Now we can deploy the model and create and endpoint that can be queried using our custom DeepARPredictor class.

# In[94]:


predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    predictor_cls=DeepARPredictor)

print(predictor)

sagemaker_endpoint = {
    "endpoint" : predictor.endpoint
}
with open("sagemaker-description.json", "w") as sagemaker_description:
    json.dump(sagemaker_endpoint, sagemaker_description)




i = 0
for n in rid_targets:
    print(i, n)

    try:
        predicted = predictor.predict(ts=timeseries[i], quantiles=[0.10, 0.5, 0.90])
        predicted.to_csv(os.path.join("./output", n + ".csv"), mode='w')
    except Exception as ex:
        print('Error occurred', ex)

    i += 1


# Below we define a plotting function that queries the model and displays the forecast.

# In[98]:


def plot(
        predictor,
        target_ts,
        cat=None,
        dynamic_feat=None,
        forecast_date=end_training,
        show_samples=False,
        plot_history=7 * 12,
        confidence=80
):
    print("calling served model to generate predictions starting from {}".format(str(forecast_date)))
    assert (confidence > 50 and confidence < 100)
    low_quantile = 0.5 - confidence * 0.005
    up_quantile = confidence * 0.005 + 0.5

    # we first construct the argument to call our model
    args = {
        "ts": target_ts[:forecast_date],
        "return_samples": show_samples,
        "quantiles": [low_quantile, 0.5, up_quantile],
        "num_samples": 100
    }

    if dynamic_feat is not None:
        args["dynamic_feat"] = dynamic_feat
        fig = plt.figure(figsize=(20, 6))
        ax = plt.subplot(2, 1, 1)
    else:
        fig = plt.figure(figsize=(20, 3))
        ax = plt.subplot(1, 1, 1)

    if cat is not None:
        args["cat"] = cat
        ax.text(0.9, 0.9, 'cat = {}'.format(cat), transform=ax.transAxes)

    # call the end point to get the prediction
    prediction = predictor.predict(**args)

    # plot the samples
    if show_samples:
        for key in prediction.keys():
            if "sample" in key:
                prediction[key].plot(color='lightskyblue', alpha=0.2, label='_nolegend_')

    # plot the target
    target_section = target_ts[forecast_date - plot_history:forecast_date + prediction_length]
    target_section.plot(color="black", label='target')

    # plot the confidence interval and the median predicted
    ax.fill_between(
        prediction[str(low_quantile)].index,
        prediction[str(low_quantile)].values,
        prediction[str(up_quantile)].values,
        color="b", alpha=0.3, label='{}% confidence interval'.format(confidence)
    )
    prediction["0.5"].plot(color="b", label='P50')
    ax.legend(loc=2)

    # fix the scale as the samples may change it
    ax.set_ylim(target_section.min() * 0.5, target_section.max() * 1.5)

    if dynamic_feat is not None:
        for i, f in enumerate(dynamic_feat, start=1):
            ax = plt.subplot(len(dynamic_feat) * 2, 1, len(dynamic_feat) + i, sharex=ax)
            feat_ts = pd.Series(
                index=pd.DatetimeIndex(start=target_ts.index[0], freq=target_ts.index.freq, periods=len(f)),
                data=f
            )
            feat_ts[forecast_date - plot_history:forecast_date + prediction_length].plot(ax=ax, color='g')


# We can interact with the function previously defined, to look at the forecast of any customer at any point in (future) time.
#
# For each request, the predictions are obtained by calling our served model on the fly.
#
# Here we forecast the consumption of an office after week-end (note the lower week-end consumption).
# You can select any time series and any forecast date, just click on `Run Interact` to generate the predictions from our served endpoint and see the plot.

# In[99]:


style = {'description_width': 'initial'}


# In[100]:


@interact_manual(
    customer_id=IntSlider(min=0, max=369, value=91, style=style),
    forecast_day=IntSlider(min=0, max=100, value=51, style=style),
    confidence=IntSlider(min=60, max=95, value=80, step=5, style=style),
    history_weeks_plot=IntSlider(min=1, max=20, value=1, style=style),
    show_samples=Checkbox(value=False),
    continuous_update=False
)
def plot_interact(customer_id, forecast_day, confidence, history_weeks_plot, show_samples):
    plot(
        predictor,
        target_ts=timeseries[customer_id],
        forecast_date=end_training + datetime.timedelta(days=forecast_day),
        show_samples=show_samples,
        plot_history=history_weeks_plot * 12 * 7,
        confidence=confidence
    )
