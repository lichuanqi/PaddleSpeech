import argparse
import json
import sys
import time
import base64

import requests

from paddlespeech.cli.log import logger
from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor


def wav2base64(wav_file: str):
    """
    read wave file and covert to base64 string
    """
    with open(wav_file, 'rb') as f:
        base64_bytes = base64.b64encode(f.read())
        base64_string = base64_bytes.decode('utf-8')
    return base64_string


def client_http_by_cli(input_wav):
    """使用cli的高级接口预测
    
    Params
        input: 音频文件

    Output
        res: 预测结果
    """
    asrclient_executor = ASRClientExecutor()
    res = asrclient_executor(
        input=input_wav,
        server_ip="192.168.35.221",
        port=8888,
        sample_rate=16000,
        lang="zh_cn",
        audio_format="wav")

    print(res)


def client_http_by_request(input_wav):

    API_ASR_HTTP = 'http://192.168.35.221:8888/paddlespeech/asr'
    
    sample_rate=16000
    lang="zh_cn"
    audio_format="wav"
    
    time_start = time.time()
    print(f"asr http client start, endpoint: {API_ASR_HTTP}")

    audio = wav2base64(input_wav)
    data = {
        "audio": audio,
        "audio_format": audio_format,
        "sample_rate": sample_rate,
        "lang": lang,
    }

    res = requests.post(API_ASR_HTTP, data=json.dumps(data))
    res = res.json()['result']['transcription']

    time_end = time.time()
    print(f"ASR http result: {res}")
    print("ASR http response time %f s." % (time_end - time_start))


if __name__=='__main__':
    input_wav = "zh.wav"

    # client_http_by_cli(input_wav)
    client_http_by_request(input_wav)
