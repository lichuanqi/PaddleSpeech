import os
from utils import MyThread
from main import request_api


def _add(a,b):
    return a+b


def thread_add():
    for a,b in zip([1,2,3], [4,5,6]):
        th = MyThread(_add, (a,b))
        th.start()
        res = th.get_result()
        print(res)


def thread_request_api():
    """测试新线程调用接口"""
    path = 'speech_client_pyqt/files/2022-12-14'
    for file in os.listdir(path):
        wavpath = os.path.join(path, file)
        
        th = MyThread(request_api, (wavpath,))
        th.start()
        res = th.get_result()
        print(res)


if __name__ == '__main__':
    # thread_add()

    thread_request_api()