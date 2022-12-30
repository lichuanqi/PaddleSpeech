"""
语音识别api接口相关函数
"""
from datetime import datetime
import json
import base64
import requests

from util import MyThread


# 接口信息
SERVER_IP = "192.168.35.221"
PORT = "8888"
ENDPOINT = "/paddlespeech/asr"

# 接口参数信息
SAMPLE_RATE = 16000
LANG = "zh_cn"
AUDIO_FORMAT = "wav"


def api_request(filepath, filedata=None):
    """根据音频文件的路径调用接口进行语音识别

    Params
        filepath: 音频文件路径
        filedata: 音频文件数据
    
    Return
        text: 文字结果
    """
    audio_format = "wav"
    sample_rate = 16000
    lang = "zh_cn"
    
    server_url = 'http://' + SERVER_IP + ":" + str(PORT) + ENDPOINT

    if filepath:
        with open(filepath, 'rb') as f:
            base64_bytes = base64.b64encode(f.read())
            audio = base64_bytes.decode('utf-8')
        
        data = {
            "audio": audio,
            "audio_format": audio_format,
            "sample_rate": sample_rate,
            "lang": lang,
        }
        
        try:
            response = requests.post(server_url, data=json.dumps(data))
            data = json.loads(response.text)
            res = data['result']['transcription']

            return res

        except requests.exceptions.ConnectionError as err:
            return err


def api_test():
    """测试接口是否连通和延迟时间

    Return
        status     : 接口连通状态
        delay_time : 接口延迟时间
    """
    server_url = 'http://' + SERVER_IP + ":" + str(PORT)

    try:
        start = datetime.now()
        response = requests.get(server_url)
        data = json.loads(response.text)
        end = datetime.now()
        time = int((end - start).total_seconds() * 1000)

        if data['success']:
            status = True
            delay_time = time

        else:
            status = False
            delay_time = time

    except Exception as err:
        status = False
        delay_time = 0

    return status, delay_time


def api_request_thread(filepath):
    """为语音识别接口新建一个线程"""
    text_threading = MyThread(api_request, (filepath, ))
    text_threading.start()
    text = text_threading.get_result()

    return text


def api_test_thread():
    """为接口测试新建一个线程"""
    text_threading = MyThread(api_test, ())
    text_threading.start()
    status, delay_time = text_threading.get_result()

    return status, delay_time


if __name__ == '__main__':
    status, delay_time = api_test_thread()
    print(status, delay_time)