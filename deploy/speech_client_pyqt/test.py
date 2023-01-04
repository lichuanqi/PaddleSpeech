import os
import sys

from PyQt5.QtWidgets import QApplication,QWidget,QKeyEventTransition
from PyQt5 import QtCore
from PyQt5.QtCore import *

from util import MyThread
from api import api_request


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
        
        th = MyThread(api_request, (wavpath,))
        th.start()
        res = th.get_result()
        print(res)


def bar_check_test():
    aa = '111111'
    print(len(aa))


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setFixedWidth(300)
        self.setFixedHeight(200)
        self.setWindowTitle('按键检测')
        self.show()
 
    # 检测键盘回车按键
    def keyPressEvent(self, event):
        print("按下：" + str(event.key()))
        # 举例
        if(event.key() == Qt.Key_Escape):
            print('测试：ESC')
        if(event.key() == Qt.Key_A):
            print('测试：A')
        if(event.key() == Qt.Key_1):
            print('测试：1')
        if(event.key() == Qt.Key_Enter):
            print('测试：Enter')
        if(event.key() == Qt.Key_Space):
            print('测试：Space')
 
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print("鼠标左键点击")
        elif event.button() == Qt.RightButton:
            print("鼠标右键点击")
        elif event.button() == Qt.MidButton:
            print("鼠标中键点击")


def pyqt_listen_key():
    """监听键盘"""
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # thread_add()

    # thread_request_api()
    # bar_check_test()

    pyqt_listen_key()