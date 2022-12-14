import os
import threading


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)
        
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


def mk_if_not_exits(path):
    if not os.path.exists(path):
        os.mkdir(path)


def list2txt(data:list, savename:str) -> None:
    """列表保存为txt文件
    Params
        data     : 要保存的列表数据
        savename : 保存文件名称
    Output
        savename.txt
    """
    with open(savename, 'w', encoding='utf-8') as f:
        for _data in data:
            if isinstance(_data, list):
                data_write = ','.join(_data) + '\n'
                f.write(data_write)
            else:
                f.write(_data)