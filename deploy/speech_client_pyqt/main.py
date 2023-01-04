import os
import sys
import time
import json
import requests

import base64
import pyaudio
import wave

from PyQt5.QtCore import *
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QDesktopWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QLabel, QTextEdit, QLineEdit
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QStatusBar

from api import api_test_thread, api_request_thread
from util import MyThread
from util import mk_if_not_exits, list2txt, get_date_time

DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.init_output_path()
        self.init_params()
        self.init_UI()
        self.init_status_bar()
        self.init_api_status()


    def init_output_path(self):
        """初始化文件保存路径"""
        savedir = os.path.join(DIR, 'files')
        mk_if_not_exits(savedir)
        date, time = get_date_time()
        self.savepath = os.path.join(savedir, f'{date}')
        mk_if_not_exits(self.savepath)


    def init_params(self):
        """初始化参数"""
        # QT参数
        self.ld_tiaoma = None
        self.table_widget = None
        self.current_row = 0

        # 录音参数
        self.recording = False
        self.start_time = 0
        self.end_time = 0
        self.record_frames = []


    def init_UI(self):
        """初始化UI界面"""
        # 设置窗体名称和尺寸
        self.setWindowTitle('语音录入系统')
        self.resize(1470, 800)
        icon_windos = QIcon('deploy/speech_client_pyqt/images/icons/logo@3x.png')
        self.setWindowIcon(icon_windos)

        # 设置窗体位置
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        # 创建总体横向布局
        layout = QVBoxLayout()

        # 条码输入区域
        tiaoma_layout = QHBoxLayout()
        ld_tiaoma = QLineEdit()
        ld_tiaoma.setClearButtonEnabled(True)
        ld_tiaoma.setPlaceholderText('请输入条码号或者扫描条码')
        ld_tiaoma.setFocus()
        ld_tiaoma.returnPressed.connect(self.start_record_threading)
        tiaoma_layout.addWidget(ld_tiaoma)
        tiaoma_layout.addStretch()
        layout.addLayout(tiaoma_layout)

        self.ld_tiaoma = ld_tiaoma

        # 录音控制区域
        kongzhi_layout = QHBoxLayout()
        bt_start = QPushButton('开始')
        icon_start = QIcon('deploy/speech_client_pyqt/images/icons/microphone@3x.png')
        bt_start.setIcon(icon_start)
        bt_start.setIconSize(QSize(21,21))
        bt_start.clicked.connect(self.start_record_threading)
        bt_end = QPushButton('结束')
        bt_end.setShortcut(QKeySequence("ALT + E"))
        icon_end = QIcon('deploy/speech_client_pyqt/images/icons/language.svg')
        bt_end.setIcon(icon_end)
        bt_end.setIconSize(QSize(24,24))
        bt_end.clicked.connect(self.stop_and_save)

        kongzhi_layout.addWidget(bt_start)
        kongzhi_layout.addWidget(bt_end)
        kongzhi_layout.addStretch()
        layout.addLayout(kongzhi_layout)
        layout.addSpacing(30)

        # 表格区域
        table_header_layout = QHBoxLayout()
        table_label = QLabel('当前已录入条目:', self)
        bt_table_clear = QPushButton('清空')
        icon_clear = QIcon('deploy/speech_client_pyqt/images/icons/delete.svg')
        bt_table_clear.setIcon(icon_clear)
        bt_table_clear.setIconSize(QSize(26,26))
        bt_table_clear.clicked.connect(self.data_clear)
        bt_table_inport = QPushButton('导入')
        icon_inport = QIcon('deploy/speech_client_pyqt/images/icons/inport.svg')
        bt_table_inport.setIcon(icon_inport)
        bt_table_inport.setIconSize(QSize(20,20))
        bt_table_inport.clicked.connect(self.data_inport)
        bt_table_export = QPushButton('导出')
        icon_export = QIcon('deploy/speech_client_pyqt/images/icons/export.svg')
        bt_table_export.setIcon(icon_export)
        bt_table_export.setIconSize(QSize(20,20))
        bt_table_export.clicked.connect(self.data_export_txt)

        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch()
        table_header_layout.addWidget(bt_table_clear)
        table_header_layout.addWidget(bt_table_inport)
        table_header_layout.addWidget(bt_table_export)
        layout.addLayout(table_header_layout)
        layout.addSpacing(10)

        # 表格内容
        table_layout =  QVBoxLayout()
        table_widget = QTableWidget(0, 5)
        table_widget.setAlternatingRowColors(True)
        table_layout.addWidget(table_widget)
        self.table_widget = table_widget

        # 表格标题
        headers = [
            {'field': 'code', 'text': '条码号', 'width': 200},
            {'field': 'address', 'text': '地址', 'width': 400},
            {'field': 'type', 'text': '类型', 'width': 100},
            {'field': 'shape', 'text': '形状', 'width': 100},
            {'field': 'filepath', 'text': '文件地址', 'width': 400},
        ]
        for i, info in enumerate(headers):
            item = QTableWidgetItem()
            item.setText(info['text'])
            table_widget.setHorizontalHeaderItem(i, item)
            table_widget.setColumnWidth(i, info['width'])
        layout.addLayout(table_layout)
        layout.addSpacing(20)

        # 底部弹簧
        # layout.addStretch()

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def init_status_bar(self):
        """初始化底部状态栏"""
        self.status_bar = QStatusBar(self)
        # 设置默认显示内容
        self.status_bar.showMessage('初始化完成')
        self.setStatusBar(self.status_bar)

    
    def init_api_status(self):
        """初始化时测试api接口的连通状态"""
        status, delay_time = api_test_thread()

        if status:
            self.status_bar.showMessage(f'接口连通, 延迟: {delay_time}ms')
        else:
            self.status_bar.showMessage(f'接口未连通')
        

    def start_record_threading(self):
        """开辟一个线程开始录音"""
        # 判断是否正在录音
        if self.recording:
            self.stop_and_save()
        
        # 获取条码号并检查格式
        else:
            tiaoma = self.ld_tiaoma.text().strip()
            res, err = check_post_id(tiaoma)
            self.post_id = tiaoma

            if not res:
                QMessageBox.warning(self, '错误', err)
            else:
                # 开始录音
                self.recording = True
                self.start_time = time.time()

                record_thread = MyThread(self.start_record)
                record_thread.start()
                self.status_bar.showMessage('正在录音中....')

                # 在表格中插入条码
                current_row = self.table_widget.rowCount()
                self.table_widget.insertRow(current_row)
                code_ = QTableWidgetItem(str(self.post_id))
                self.table_widget.setItem(current_row, 0, code_)
                self.current_row = current_row

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
        # 判断是否正在录音
        if not self.recording:
            QMessageBox.warning(self, '错误', '请先开始录音')

        else:
            # 判断时间
            self.end_time = time.time()
            audio_time = self.end_time - self.start_time
            if audio_time <= 3:
                QMessageBox.warning(self, '警告', '录音小于3s, 请重新开始')
                self.status_bar.showMessage(f'录音非正常结束')

            else:
                # 文件保存名称
                savename = os.path.join(self.savepath, f'{self.post_id}.wav')

                # 保存音频文件
                p = pyaudio.PyAudio()
                wf = wave.open(savename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(self.record_frames))
                wf.close()

                # 在表格中插入音频文件的保存名称
                filepath = savename.replace(DIR, '')
                item_filepath = QTableWidgetItem(str(filepath))
                self.table_widget.setItem(self.current_row, 4, item_filepath)
                self.status_bar.showMessage(f'录音正常结束, 文件保存至: {filepath}')
                
                # 预测结果
                self.api_predict_threading(filepath)
            
            # 录音参数重置
            self.recording = False
            self.record_frames = []


    def api_predict_threading(self, audio_file):
        """开辟一个线程调用接口预测"""
        # 判断是否正在录音
        if not self.recording:
            QMessageBox.warning(self, '错误', '请先开始录音')
        
        else:
            # 调用接口预测
            text = api_request_thread(audio_file)

            # 预测结束后在表格中插入地址
            address = QTableWidgetItem(str(text))
            self.table_widget.setItem(self.current_row, 1, address)

            self.status_bar.showMessage(f'预测完成: {text}')

            self.current_row = None


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

        self.status_bar.showMessage(f'数据已导入')


    def data_export_txt(self):
        """将表格中的数据导出到txt文件"""
        table_data = []
        table_count = self.table_widget.rowCount()

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

            self.status_bar.showMessage(f'数据导出至: {filepath}')
        else:
            QMessageBox.warning(self, '警告', '当前表格区域无数据可导出')


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
        err = f'条码非纯数字, 请检查: {post_id}'
    elif len(post_id) <= 5:
        res = False
        err = f'条码号长度小于5, 请检查: {post_id}'
    else:
        res, err = True, 'None'

    return res, err


if __name__ == '__main__':
    # 设置字体
    font = QFont()
    font.setFamily('宋体')
    font.setPointSize(12)

    app = QApplication(sys.argv)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())