import argparse
import json
import sys
import time

import requests

from paddlespeech.cli.log import logger
from paddlespeech.server.utils.util import wav2base64

from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor


def client_cli():
    """
    使用cli的高级接口预测
    """
    asrclient_executor = ASRClientExecutor()
    res = asrclient_executor(
        input="/home/xxtc/lichuan/Dataset/speech/002.wav",
        server_ip="127.0.0.1",
        port=8090,
        sample_rate=16000,
        lang="zh_cn",
        audio_format="wav")

    print(res)


def client_request():
    """使用request库"""
    input="zh.wav"
    server_ip="127.0.0.1"
    port="8899"
    sample_rate=16000
    lang="zh_cn"
    audio_format="wav"
    endpoint="/paddlespeech/asr"
    protocol = "http"

    time_start = time.time()

    if protocol.lower() == "http":
        
        if server_ip is None or port is None:
            server_url = None
        else:
            server_url = 'http://' + server_ip + ":" + str(port) + endpoint
        logger.info(f"asr http client start, endpoint: {server_url}")

        audio = wav2base64(input)
        data = {
            "audio": audio,
            "audio_format": audio_format,
            "sample_rate": sample_rate,
            "lang": lang,
        }

        res = requests.post(server_url, data=json.dumps(data))
        res = res.json()['result']['transcription']
        
    else:
        logger.error(f"Sorry, we have not support protocol: {protocol},"
                        "please use http or websocket protocol")
        sys.exit(-1)

    time_end = time.time()
    logger.info("asr http client finished")
    logger.info(f"ASR result: {res}")
    logger.info("Response time %f s." % (time_end - time_start))


if __name__=='__main__':
    client_request()