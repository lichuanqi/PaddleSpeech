import os
import sys
from datetime import datetime
from loguru import logger 


class log():
    def __init__(self, stdout=False, savepath=False) -> None:
        self.logger = logger
        # 清空所有设置
        self.logger.remove()

        if stdout:
            # 日志输出到控制台
            self.logger.add(sys.stdout)

        if savepath:
            # 如果存在保存路径就生成日志文件
            logname =  savepath + 'log_{time}.log'
            self.logger.add(logname)


def mk_if_not_exits(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_time():
    """
    获取当前时间
    """
    ct = datetime.now()
    time_str = ct.strftime(r'%Y-%m-%d %H:%M:%S')

    return time_str
    # return ct.year, ct.month, ct.day, ct.hour, ct.minute


if __name__ == '__main__':
    print(get_time())