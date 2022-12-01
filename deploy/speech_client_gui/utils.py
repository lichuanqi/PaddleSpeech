import os


def mk_if_not_exits(path):
    if not os.path.exists(path):
        os.mkdir(path)