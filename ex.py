# -*- coding: utf-8 -*-

# lambda_function.py

from __future__ import print_function

import os
import re
import urllib
import boto3
import cv2

event = {
    "Records": [
        {
            "eventVersion": "2.1",
            "eventSource": "aws:s3",
            "awsRegion": "ap-northeast-1",
            "eventTime": "2019-07-04T14:19:32.997Z",
            "eventName": "ObjectCreated:Put",
            "userIdentity": {
                "principalId": "AWS:AIDASD57JRPQ3DMJDY23M"
            },
            "requestParameters": {
                "sourceIPAddress": "217.178.82.135"
            },
            "responseElements": {
                "x-amz-request-id": "04CAAD6D44C3A531",
                "x-amz-id-2": "5fu+qLCqyQciYBf49iVRNVi3fE8if8vI+jV+eJ4ioSvGYk4smGyWzOx1bb7+zwrnTtrQIxN72q4="
            },
            "s3": {
                "s3SchemaVersion": "1.0",
                "configurationId": "ada661c7-31c3-4326-b6d3-12b1d4571ef0",
                "bucket": {
                    "name": "temporary-image.intv.co.jp",
                    "ownerIdentity": {
                        "principalId": "A1L6HOT97DTEPI"
                    },
                    "arn": "arn:aws:s3:::temporary-image.intv.co.jp"
                },
                "object": {
                    "key": "03.jpg",
                    "size": 137216,
                    "eTag": "2e4d70b234f6632aefe00eee18950d07",
                    "sequencer": "005D1E0AF4DA79CD34"
                }
            }
        }
    ]
}

s3 = boto3.resource('s3')

# 正面顔検出用学習データを読み込み
cascade = cv2.CascadeClassifier('/s3-trigger/haarcascade_frontalface_default.xml')

def lambda_handler(event, context=' '):
    # S3のファイル情報を取得/設定
    bucket = s3.Bucket(event['Records'][0]['s3']['bucket']['name'])
    key = 'input-data/' + urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    tmp = '/tmp/' + os.path.basename(key)
    output = re.sub('input-data/', 'output-data/', key)
    try:
        # 処理前の画像を読み込み
        bucket.download_file(key=key, filename=tmp)
        before = cv2.imread(tmp)
        after = cv2.imread(tmp)

        # グレースケールに変換
        gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        # カスケード型分類器で顔を検出
        face = cascade.detectMultiScale(gray, 1.3, 5)
        if 0 < len(face):
            for (x, y, w, h) in face:
                # 顔を切り取りモザイク処理
                cut_img = before[y:y + h, x:x + w]
                cut_face = cut_img.shape[:2][::-1]
                cut_img = cv2.resize(cut_img, (cut_face[0] / 10, cut_face[0] / 10))
                cut_img = cv2.resize(cut_img, cut_face, interpolation=cv2.INTER_NEAREST)
                # モザイク処理した部分を重ねる
                after[y:y + h, x:x + w] = cut_img

        # 処理後の画像を保存
        cv2.imwrite(tmp, after)
        s3.upload_file(Filename=tmp, Bucket=bucket, Key=output)
        os.remove(tmp)
        return
    except Exception as e:
        print(e)
        raise e

lambda_handler(event)