import os

import boto3
import numpy as np
import pandas as pd

bucket = "smedu-20201114-std25"   # Modify for you
prefix = "/"     # Modify for you

boto3.setup_default_session(profile_name='default')

client = boto3.client('s3')


def get_files(path_prefix):
    result = []

    paginator = client.get_paginator('list_objects')
    operation_parameters = {'Bucket': bucket,
                            'Prefix': path_prefix}
    for file in paginator.paginate(**operation_parameters):
        if file.get("Contents", None) is None:
            continue

        for content in file['Contents']:
            path = content['Key']

            if len(path) - 1 == path.rfind('/'):
                continue

            result.append(content['Key'])

    return result


files = get_files(prefix)
print(files)


def get_demands(path_prefix):
    result = []

    for p in get_files(path_prefix):
        if p.find(".csv") == -1:
            continue

        # rId = os.path.splitext(os.path.basename(p))[0]

        obj = client.get_object(Bucket=bucket, Key=p)

        if not result:
            result.append(pd.read_csv(obj['Body']))

    frame = pd.concat(result, axis=0, ignore_index=True)

    return frame


demands = get_demands(prefix)
print(demands)
