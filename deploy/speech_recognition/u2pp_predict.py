import os
import glob
import sys
import time
import paddle
from paddlespeech.cli.asr.infer import ASRExecutor

# 参数
model='conformer_u2pp_online_wenetspeech'
lang='zh'
codeswitch=False
sample_rate=16000
config=None
decode_method='attention_rescoring'
num_decoding_left_chunks=-1
ckpt_path=None
force_yes=True
audio_file = 'D:/CODE/CPRI_ASR_Register/client/files/2023-03-01/'

# 模型初始化
print('开始初始化')
t_start = time.time()
asr_executor = ASRExecutor()
asr_executor._init_from_path(
    model_type=model, 
    lang=lang, 
    sample_rate=sample_rate, 
    cfg_path=config,
    decode_method=decode_method,
    num_decoding_left_chunks=num_decoding_left_chunks, 
    ckpt_path=ckpt_path)
paddle.set_device(paddle.get_device())
t_end = time.time()
print('初始化完成，用时: %s'%(t_end-t_start))

# 判断文件夹还是文件
files = []
if os.path.isdir(audio_file):
        files = glob.glob(audio_file + '*.wav')
elif 'txt' in audio_file:
    with open(audio_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_str = line.strip('\n').split('\t')
            files.append(line_str[0])
else:
    files = [audio_file]
num = len(files)
print('读取到 {} 个语音文件'.format(num))

# 开始逐个文件预测
for i in range(num):
    file = files[i]
    t_start = time.time()
    
    # 检查文件格式
    if not asr_executor._check(file, sample_rate, force_yes):
        sys.exit(-1)
    
    asr_executor.preprocess(model, file)
    asr_executor.infer(model)
    res = asr_executor.postprocess()
    t_end = time.time()
    print('第%s个预测完成，用时: %s'%(i, t_end-t_start))
    print('ASR Result: \n{}'.format(res))