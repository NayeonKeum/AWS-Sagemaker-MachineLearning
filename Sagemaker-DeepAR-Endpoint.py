from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON
import json
import os
import boto3
import sagemaker
import numpy as np
import pandas as pd
import ntpath

bucket = "vpp-export"

s3_bucket = 'vpp-export'  # replace with an existing bucket if needed
s3_prefix = 'v1'  # prefix used for all data stored within the bucket

sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::022399919362:role/MeterForecast-SageMakerIamRole-11Y2RB97CKMRM"
region = boto3.Session().region_name
endpoint_name = "deepar-electricity-demo-2019-12-12-06-30-51-282"

runtime = boto3.Session().client('sagemaker-runtime')
content_type = 'text/csv'

s3_output_path = "s3://{}/{}/out".format(s3_bucket, s3_prefix)

freq = '1H'

local_data_dir_path = "./data/"
rid_targets = {}

endpoint_description = "sagemaker-description.json"
if os.path.isfile(endpoint_description):
    with open(endpoint_description, "r") as st_json:
        sagemaker_endpoint = json.load(st_json)
        print(sagemaker_endpoint)
        endpoint_name = sagemaker_endpoint.endpoint


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
        data = data.sort_values(by='Timestamp', ascending=False)
        data = data.iloc[0:1000]

        data_kw = data.resample(freq).sum()

        if j == 0:
            timeseries.append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))
        else:
            timeseries[i] = timeseries[i].append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))

        j += 1
    i += 1

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, content_type=sagemaker.content_types.CONTENT_TYPE_JSON, **kwargs)

    def predict(self, ts, cat=None, dynamic_feat=None,
                num_samples=100, return_samples=False, quantiles=["0.1", "0.5", "0.9"]):
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
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat
    return obj


predictor = DeepARPredictor(
    endpoint=endpoint_name,
    sagemaker_session=sagemaker_session)


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



i = 0
for n in rid_targets:
    print(i, n)

    try:
        predicted = predictor.predict(ts=timeseries[i], quantiles=[0.10, 0.5, 0.90])

        csv_file_name = n + ".csv"
        csv_file_path = os.path.join("./output", csv_file_name)

        predicted.to_csv(csv_file_path, mode='w')

        copy_to_s3(csv_file_path, s3_output_path + "/" + csv_file_name, override=True)

    except Exception as ex:
        print('Error occured', ex)
    i += 1


