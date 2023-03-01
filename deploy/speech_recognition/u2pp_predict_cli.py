import time
import paddle
from paddlespeech.cli.asr.infer import ASRExecutor

t_start = time.time()
print('开始初始化')
asr_executor = ASRExecutor()
t_end = time.time()
print('初始化完成，用时: %s'%(t_end-t_start))

t_start = time.time()
text = asr_executor(
    model='conformer_u2pp_online_wenetspeech',
    lang='zh',
    sample_rate=16000,
    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
    ckpt_path=None,
    audio_file='zh.wav',
    force_yes=False,
    device=paddle.get_device())
t_end = time.time()
print('预测完成，用时: %s'%(t_end-t_start))
print('ASR Result: \n{}'.format(text))