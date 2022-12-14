import os
import sys
import json
import requests

import base64
import pyaudio
import wave

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QLabel, QTextEdit, QLineEdit
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from utils import MyThread, mk_if_not_exits, list2txt
from utils import get_date_time

DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        # 初始化文件保存路径
        savedir = os.path.join(DIR, 'files')
        mk_if_not_exits(savedir)
        date, time = get_date_time()
        self.savepath = os.path.join(savedir, f'{date}')
        mk_if_not_exits(self.savepath)
        
        self.previous_file = None
        self.current_row = 0

        # QT
        self.ld_tiaoma = None
        self.table_widget = None

        # 录音参数
        self.recording = False
        self.record_frames = []

        self.init_UI()


    def init_UI(self):
        """UI界面"""
        # 设置窗体名称和尺寸
        self.setWindowTitle('语音录入系统')
        self.resize(1470, 800)

        # 设置窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 创建布局
        layout = QVBoxLayout()
        
        # 顶部菜单
        menue_layout = QHBoxLayout()
        bt_luru = QPushButton('录入')
        bt_setting = QPushButton('设置')
        menue_layout.addWidget(bt_luru)
        menue_layout.addWidget(bt_setting)
        menue_layout.addStretch()
        layout.addLayout(menue_layout)

        # 条码输入区域
        tiaoma_layout = QHBoxLayout()
        ld_tiaoma = QLineEdit()
        ld_tiaoma.setPlaceholderText('请输入条码号或者扫描条码')
        tiaoma_layout.addWidget(ld_tiaoma)
        tiaoma_layout.addStretch()
        layout.addLayout(tiaoma_layout)

        self.ld_tiaoma = ld_tiaoma

        # 语音控制
        kongzhi_layout = QHBoxLayout()
        bt_start = QPushButton('开始')
        bt_start.clicked.connect(self.start_record_threading)
        bt_end = QPushButton('结束')
        bt_end.clicked.connect(self.stop_and_save)
        bt_predict = QPushButton('预测')
        bt_predict.clicked.connect(self.api_predict_threading)
        kongzhi_layout.addWidget(bt_start)
        kongzhi_layout.addWidget(bt_end)
        kongzhi_layout.addWidget(bt_predict)
        kongzhi_layout.addStretch()
        layout.addLayout(kongzhi_layout)

        # 表格标题
        table_header_layout = QHBoxLayout()
        table_label = QLabel('当前已录入条目:', self)
        bt_table_clear = QPushButton('清空')
        bt_table_clear.clicked.connect(self.data_clear)
        bt_table_inport = QPushButton('导入')
        bt_table_inport.clicked.connect(self.data_inport)
        bt_table_export = QPushButton('导出')
        bt_table_export.clicked.connect(self.data_export_txt)

        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch()
        table_header_layout.addWidget(bt_table_clear)
        table_header_layout.addWidget(bt_table_inport)
        table_header_layout.addWidget(bt_table_export)
        layout.addLayout(table_header_layout)

        # 表格区域
        table_layout =  QVBoxLayout()
        table_widget = QTableWidget(0, 5)
        table_layout.addWidget(table_widget)
        self.table_widget = table_widget

        # 设置表格表头
        headers = [
            {'field': 'code', 'text': '条码号', 'width': 300},
            {'field': 'address', 'text': '地址', 'width': 400},
            {'field': 'type', 'text': '类型', 'width': 150},
            {'field': 'shape', 'text': '形状', 'width': 150},
            {'field': 'filepath', 'text': '文件地址', 'width': 400},
        ]
        for i, info in enumerate(headers):
            item = QTableWidgetItem()
            item.setText(info['text'])
            table_widget.setHorizontalHeaderItem(i, item)
            table_widget.setColumnWidth(i, info['width'])
        layout.addLayout(table_layout)

        # 日志情况
        log_layout =  QVBoxLayout()
        log_label = QLabel('日志', self)
        log_text = QTextEdit()
        log_layout.addWidget(log_label)
        log_layout.addWidget(log_text)
        layout.addLayout(log_layout)

        # 底部弹簧
        # layout.addStretch()

        self.setLayout(layout)


    def start_record_threading(self):
        """开辟一个线程开始录音"""
        # 获取条码号并检查格式
        tiaoma = self.ld_tiaoma.text().strip()
        res, err = check_post_id(tiaoma)
        self.post_id = tiaoma

        if res:
            # 开始录音时在表格中插入条码
            current_row = self.table_widget.rowCount()
            self.table_widget.insertRow(current_row)
            code_ = QTableWidgetItem(str(self.post_id))
            self.table_widget.setItem(current_row, 0, code_)
            self.current_row = current_row

            # 开始录音
            record_thread = MyThread(self.start_record)
            record_thread.start()
        else:
            QMessageBox.warning(self, '错误', '条码格式问题')

        # 条码框清空
        self.ld_tiaoma.clear()


    def start_record(self):
        """开始录音
        """
        self.recording = True
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

        while self.recording:
            data = stream.read(1024)
            self.record_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()


    def stop_and_save(self):
        """停止录音,将录音文件保存至本地指定路径
        """
        # 文件保存名称
        savename = os.path.join(self.savepath, f'{self.post_id}.wav')

        if self.recording:
            p = pyaudio.PyAudio()
            wf = wave.open(savename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.record_frames))
            wf.close()

            # 结束录音时在表格中插入文件地址
            filepath = savename.replace(DIR, '')
            item_filepath = QTableWidgetItem(str(filepath))
            self.table_widget.setItem(self.current_row, 4, item_filepath)

            self.previous_file = savename
            self.recording = False
            self.record_frames = []

        else:
            QMessageBox.warning(self, '错误', '请先开始录音')


    def api_predict_threading(self):
        """开辟一个线程调用接口预测"""
        if self.previous_file:
            text_threading = MyThread(request_api, (self.previous_file, ))
            text_threading.start()
            text = text_threading.get_result()

            # 预测结束后在表格中插入地址
            address = QTableWidgetItem(str(text))
            self.table_widget.setItem(self.current_row, 1, address)

            self.current_row = None

        else:
            QMessageBox.warning(self, '错误', '请先开始录音')


    def data_clear(self):
        """清空表格中的数据"""
        self.table_widget.setRowCount(0)
        self.table_widget.clearContents()


    def data_inport(self):
        """将txt文件中的数据导入到表格"""
        # 选择文件位置
        data_path, data_type = QFileDialog.getOpenFileName(self,  "选取文件","./files", 
            "Text Files (*.txt);;All Files (*)")

        # 文件夹不存在停止
        if not os.path.exists(data_path):
            return  ''
        else:
            with open(data_path, mode='r', encoding='utf-8') as f:
                data = f.readlines()
        
        # 数据插入到表格
        table_count = self.table_widget.rowCount()
        for row in data:
            self.table_widget.insertRow(table_count)
            code, address, typ, shape, filepath = row.strip().split(',')

            code_ = QTableWidgetItem(str(code))
            self.table_widget.setItem(table_count, 0, code_)
            address_ = QTableWidgetItem(str(address))
            self.table_widget.setItem(table_count, 1, address_)
            typ_ = QTableWidgetItem(str(typ))
            self.table_widget.setItem(table_count, 2, typ_)
            shape_ = QTableWidgetItem(str(shape))
            self.table_widget.setItem(table_count, 3, shape_)
            filepath_ = QTableWidgetItem(str(filepath))
            self.table_widget.setItem(table_count, 4, filepath_)
            table_count += 1


    def data_export_txt(self):
        """将表格中的数据导出到txt文件"""
        table_data = []
        table_count = self.table_widget.rowCount()
        print(table_count)
        for row in range(table_count):
            line = []
            for col in range(5):
                tab_item = self.table_widget.item(row, col)
                if tab_item == None:
                    line.append('')
                else:
                    line.append(str(tab_item.text()))
            table_data.append(line)

        # 数据为零
        if len(table_data) >= 1:
            filepath, pathtype = QFileDialog.getSaveFileName(self, "文件保存", 
                    f"{self.savepath}/data_list.txt" ,'txt(*.txt)')
            list2txt(table_data, filepath)
        else:
            QMessageBox.warning(self, '警告', '当前表格无数据')


def check_post_id(post_id: str):
    """检查post_id是否符合要求

    检查顺序:
        1.是否为默认空值
        2.是否为纯数字
        3.TODO: 长度是否符合

    Params
        post_id (str)
    Return
        res (bool): 检查结果,通过检查为True
        err (str) : 检查结果为False时的错误提示
    """
    if post_id == '':
        res = False
        err = '条码为空,请先输入条码'
    elif not post_id.isdigit():
        res = False
        err = f'条码非纯数字,请检查: {post_id}'
    else:
        res, err = True, 'None'

    return res, err


def request_api(filepath, filedata=None):
    """根据音频文件的路径调用接口进行语音识别

    Params
        filepath: 音频文件路径
        filedata: 音频文件数据
    Return
        text:
    """
    # 接口信息
    server_ip = "192.168.35.221"
    # server_ip = "127.0.0.1"
    port = "8888"
    sample_rate = 16000
    lang = "zh_cn"
    audio_format = "wav"
    endpoint = "/paddlespeech/asr"

    server_url = 'http://' + server_ip + ":" + str(port) + endpoint

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())