import time
import paddle
from paddlespeech.cli.text import TextExecutor


t_start = time.time()
print('开始初始化')
text_executor = TextExecutor()
t_end = time.time()
print('初始化完成，用时: %s'%(t_end-t_start))

t_start = time.time()
result = text_executor(
    text='今天的天气真不错啊你下午有空吗我想约你一起去吃饭',
    task='punc',
    model='ernie_linear_p7_wudao',
    lang='zh',
    config='models/ernie_linear_p7_wudao-punc-zh_v1.0/ckpt/config.json',
    ckpt_path='models/ernie_linear_p7_wudao-punc-zh_v1.0/ckpt/model_state.pdparams',
    punc_vocab='models/ernie_linear_p7_wudao-punc-zh_v1.0/punc_vocab.txt',
    device=paddle.get_device())
t_end = time.time()
print('预测完成，用时: %s'%(t_end-t_start))
print('Text Result: \n{}'.format(result))