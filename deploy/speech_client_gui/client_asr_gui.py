import os
import sys
import time
import json
import base64
import requests

import pyaudio
from winsound import PlaySound
import wave

import threading
import tkinter

path = os.getcwd()
sys.path.append(path)
from utils import mk_if_not_exits


class Recorder:
    def __init__(self, chunk=1024, channels=1, rate=16000):
        # 录音参数
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = False
        self._frames = []

        self.init_output()
        self.previous_path = "zh.wav"

    def init_output(self):
        self.path_file = os.path.join(path, 'files')
        self.path_wavs = os.path.join(path, 'files', 'wavs')
        self.path_results = os.path.join(path, 'files', 'results')

        mk_if_not_exits(self.path_file)
        mk_if_not_exits(self.path_wavs)
        mk_if_not_exits(self.path_results)

    def UI(self):
        # 创建窗口，设置大小
        window = tkinter.Tk()
        window.title("语音识别客户端 Demo")
        window.geometry("600x400")

        font = ("FangSong", 14)

        # 条码
        b1 = tkinter.Button(window, text="扫描条码", font=font, width=9, height=1)
        b1.place(x=10, y=10, anchor="nw")
        # 条码输入框
        e1_text = tkinter.StringVar()
        e1 = tkinter.Entry(window, bd=4, width=30, textvariable=e1_text, font=font)
        e1.place(x=120, y=12)
        self.e1 = e1
        
        # 录音按钮
        b1 = tkinter.Button(window, text="开始录音", font=font, width=9, height=1, command=self.start)
        b1.place(x=10, y=50, anchor="nw")
        # 停止按钮
        b2 = tkinter.Button(window, text="结束并保存", font=font, width=10, height=1, command=self.stop_save)
        b2.place(x=110, y=50, anchor="nw")
        # 播放按钮
        b4 = tkinter.Button(window, text="播放录音", font=font, width=9, height=1, command=self.play)
        b4.place(x=250, y=50, anchor="nw")
        # http预测按钮
        b5 = tkinter.Button(window, text="http预测", font=font, width=9, height=1, command=self.http_predict_thread)
        b5.place(x=380, y=50, anchor="nw")

        # 输出结果文本框
        self.result_label = tkinter.Label(window, text="输出日志：", font=font)
        self.result_label.place(x=10, y=110)
        self.result_text = tkinter.Text(window, width=75, height=20)
        self.result_text.place(x=10, y=130)

        # 主窗口循环
        window.mainloop()

    def get_current_time(self):
        """
        获取当前时间
        """
        current_time = time.strftime('%H:%M:%S',time.localtime(time.time()))
        return current_time

    def start(self):
        """
        为开始录音开辟一个线程
        """
        threading._start_new_thread(self._recording, ())

    def _recording(self):
        """
        开始录音
        """
        # 检查邮件条码号是否符合要求
        post_id = self.e1.get()
        res, err = check_post_id(str(post_id))
        if not res:
            self.result_text.insert('end', err)

        else:
            msg = f'识别到条码号: {post_id}\n'
            self.result_text.insert('end', msg)
            self.post_id_checked = post_id

            self._running = True
            msg = self.get_current_time() + " 开始录音...\n"
            self.result_text.insert('end', msg)

            self._frames = []
            p = pyaudio.PyAudio()

            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)

            while self._running:
                data = stream.read(self.CHUNK)
                self._frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()


    def stop_save(self):
        """
        停止录音,将录音文件保存至本地指定路径
        """
        # 保存路径及名称
        savename = os.path.join(self.path_wavs, f'{self.post_id_checked}.wav')

        if self._running:
            # 录音结束
            self._running = False

            # 保存
            p = pyaudio.PyAudio()
            wf = wave.open(savename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self._frames))
            wf.close()
            self.previous_path = savename
            msg = self.get_current_time() + f" 录音结束,保存至 {savename}\n"
            self.result_text.insert('end', msg)

        else:
            msg = self.get_current_time() + ' 请先开始录音\n'
            self.result_text.insert('end', msg)


    def play(self):
        """
        根据文件保存路径播放音频
        """
        if self.previous_path:
            PlaySound(self.previous_path, flags=1)
            msg = self.get_current_time() + " 正在播放文件: {}\n".format(self.previous_path)
        else:
            msg = self.get_current_time() + " 请先录音并保存"
        self.result_text.insert('end', msg)

    def http_predict_thread(self):
        """
        预测线程
        """
        threading._start_new_thread(self.http_predict, ())

    def http_predict(self):
        """
        根据文件保存路径使用http接口进行语音识别
        """
        # 检查邮件条码号是否符合要求
        post_id = self.e1.get()
        res, err = check_post_id(str(post_id))
        if not res:
            self.result_text.insert('end', err)

        else:
            server_ip="192.168.35.221"
            # server_ip="127.0.0.1"
            port="8888"
            sample_rate=16000
            lang="zh_cn"
            audio_format="wav"
            endpoint="/paddlespeech/asr"

            input = self.previous_path
            msg = self.get_current_time() + " 开始预测语音文件：%s\n" % (input)
            self.result_text.insert('end', msg)
            
            # 音频文件编码
            with open(input, 'rb') as f:
                base64_bytes = base64.b64encode(f.read())
                audio = base64_bytes.decode('utf-8')
            
            # 拼接api
            if server_ip is None or port is None:
                server_url = None
            else:
                server_url = 'http://' + server_ip + ":" + str(port) + endpoint
            msg = self.get_current_time() + f" 调用接口: {server_url}\n"
            self.result_text.insert('end', msg)

            data = {
                "audio": audio,
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "lang": lang,
            }
            
            try:
                # 调用接口
                re_ts = time.time()
                response = requests.post(server_url, data=json.dumps(data))
                data = json.loads(response.text)
                re_tn = time.time()
                msg = self.get_current_time() + " 接口响应时间: %.2f s.\n" % (re_tn - re_ts)
                self.result_text.insert('end', msg)

                res = data['result']['transcription']
                msg = self.get_current_time() +  f" 语音识别结果: {res}\n"
                self.result_text.insert('end', msg)

                # 结果保存至txt
                self.result_to_txt(post_id, res)

            # 无法连接接口
            except requests.exceptions.ConnectionError as err:
                msg = self.get_current_time()  + str(err) + '\n'
                self.result_text.insert('end', msg)


    def result_to_txt(self, id, result):
        """把识别结果保存为txt文件
        """
        savename = os.path.join(self.path_results, f'{self.post_id_checked}.txt')
        with open(savename, 'w', encoding='utf-8') as f:
            f.write(id + ' ' + result)
        
        msg = self.get_current_time() + f'结果已保存至: {savename}'
        self.result_text.insert('end', msg)


def check_post_id(post_id: str):
    """检查post_id是否符合要求

    检查顺序:
        (1)是否为默认空值
        (2)是否为纯数字
        (3) TODO: 长度是否符合

    Params
        post_id (str)
    Return
        res (bool): 检查结果,通过检查为True
        err (str) : 检查结果为False时的错误提示
    """
    if post_id =='':
        res = False
        err = '条码为空,请先输入条码\n'
    elif not post_id.isdigit():
        res = False
        err = f'条码不为纯数字,请检查: {post_id}\n'
    else:
        res, err = True, 'None'

    return res, err

if __name__=='__main__':
    re = Recorder()
    re.UI()