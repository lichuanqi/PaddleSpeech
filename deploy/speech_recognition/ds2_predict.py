# 语音文件的语音识别ASR,使用低级接口实现
# 预测单个wav音频文件/文件夹内所有文件
# 使用网络参数config文件和权重pdparams文件预测

from math import hypot
import os
import glob
import sys
import time
import io
import base64
import argparse
from pathlib import Path
import librosa
import numpy as np

import soundfile
from yacs.config import CfgNode

import paddle
from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
# 评价指标计算
from paddlespeech.s2t.utils import error_rate
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.utility import print_arguments
from paddlespeech.s2t.utils.utility import UpdateConfig
logger = Log(__name__).getlog()

# TODO(hui zhang): dynamic load


class DeepSpeech2Tester_hub():
    def __init__(self, config, ngpu):
        self.config = config
        self.ngpu = ngpu

        self.preprocess_conf = config.preprocess_config
        self.preprocess_args = {"train": False}
        self.preprocessing = Transformation(self.preprocess_conf)

        self.text_feature = TextFeaturizer(
            unit_type=config.unit_type,
            vocab=config.vocab_filepath,
            spm_model_prefix=config.spm_model_prefix)
        paddle.set_device('gpu' if self.ngpu > 0 else 'cpu')

    def compute_result_transcripts(self, audio, audio_len, vocab_list, cfg):
        decode_batch_size = cfg.decode_batch_size
        self.model.decoder.init_decoder(
            decode_batch_size, vocab_list, cfg.decoding_method,
            cfg.lang_model_path, cfg.alpha, cfg.beta, cfg.beam_size,
            cfg.cutoff_prob, cfg.cutoff_top_n, cfg.num_proc_bsearch)
        result_transcripts = self.model.decode(audio, audio_len)
        return result_transcripts

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self, audio_file):
        self.model.eval()
        self.audio_file = audio_file

        audio, sample_rate = soundfile.read(
            self.audio_file, dtype="int16", always_2d=True)

        audio = audio[:, 0]
        logger.info(f"audio shape: {audio.shape}")

        # fbank
        feat = self.preprocessing(audio, **self.preprocess_args)
        logger.info(f"feat shape: {feat.shape}")

        audio_len = paddle.to_tensor(feat.shape[0])
        audio = paddle.to_tensor(feat, dtype='float32').unsqueeze(axis=0)

        result_transcripts = self.compute_result_transcripts(
            audio, audio_len, self.text_feature.vocab_list, self.config.decode)

        logger.info("result_transcripts: " + result_transcripts[0])

        return result_transcripts[0]


    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test_wav_data(self, audio_data):
        self.model.eval()
        self.audio_data = audio_data

        # 解码
        decode_ts = time.time()
        audio_decode = base64.b64decode(self.audio_data)
        decode_te = time.time()
        logger.info("base64 decode time %f s." % (decode_te - decode_ts))

        # 读取到内存
        audio_bytes = io.BytesIO(audio_decode)
        audio, audio_sample_rate = soundfile.read(audio_bytes, dtype="int16", always_2d=True)

        if audio.shape[1] >= 2:
            audio = audio.mean(axis=1, dtype=np.int16)
        else:
            audio = audio[:, 0]
        # pcm16 -> pcm 32
        audio = _pcm16to32(audio)
        audio = librosa.resample(
            audio,
            orig_sr=audio_sample_rate,
            target_sr=16000)
        # pcm32 -> pcm 16
        audio = _pcm32to16(audio)

        logger.info(f"audio shape: {audio.shape}")

        # fbank
        feat = self.preprocessing(audio, **self.preprocess_args)
        logger.info(f"feat shape: {feat.shape}")

        audio_len = paddle.to_tensor(feat.shape[0])
        audio = paddle.to_tensor(feat, dtype='float32').unsqueeze(axis=0)

        result_transcripts = self.compute_result_transcripts(
            audio, audio_len, self.text_feature.vocab_list, self.config.decode)

        logger.info("result_transcripts: " + result_transcripts[0])

        return result_transcripts[0]

    def run_test(self, audio_file):
        self.audio_file = audio_file
        self.resume()
        try:
            self.test(self.audio_file)
        except KeyboardInterrupt:
            exit(-1)

    def setup_model(self):
        config = self.config.clone()
        with UpdateConfig(config):
            config.input_dim = config.feat_dim
            config.output_dim = self.text_feature.vocab_size
        model = DeepSpeech2Model.from_config(config)
        self.model = model

    def setup_checkpointer(self):
        """Create a directory used to save checkpoints into.

        It is "checkpoints" inside the output directory.
        """
        # checkpoint dir
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

        self.checkpoint = Checkpoint(
            kbest_n=self.config.checkpoint.kbest_n,
            latest_n=self.config.checkpoint.latest_n)

    def resume(self,ckpt_path):
        """Resume from the checkpoint at checkpoints in the output
        directory or load a specified checkpoint.
        """
        self.ckpt_path = ckpt_path
        params_path = self.ckpt_path + ".pdparams"
        model_dict = paddle.load(params_path)
        self.model.set_state_dict(model_dict)


def _pcm16to32(audio):
    assert (audio.dtype == np.int16)
    audio = audio.astype("float32")
    bits = np.iinfo(np.int16).bits
    audio = audio / (2**(bits - 1))
    return audio


def _pcm32to16(audio):
    assert (audio.dtype == np.float32)
    bits = np.iinfo(np.int16).bits
    audio = audio * (2**(bits - 1))
    audio = np.round(audio).astype("int16")
    return audio


def check(audio_file):
    if not os.path.isfile(audio_file):
        print("Please input the right audio file path")
        sys.exit(-1)

    logger.info("checking the audio file format......")
    try:
        sig, sample_rate = soundfile.read(audio_file)
    except Exception as e:
        logger.error(str(e))
        logger.error(
            "can not open the wav file, please check the audio file format")
        sys.exit(-1)
    logger.info("The sample rate is %d" % sample_rate)
    assert (sample_rate == 16000)
    logger.info("The audio file format is right")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, default=1, help="number of parallel processes. 0 for cpu.")
    parser.add_argument('--nxpu',type=int, default=0, choices=[0, 1], help="if nxpu == 0 and ngpu == 0, use cpu.")
    parser.add_argument("--config", 
                # default='configs/conformer_wenetspeech-zh-16k_1.0/conf/conformer.yaml',
                default='models/deepspeech2_online_wenetspeech_1.0.4/conf/deepspeech2_online.yaml',
                metavar="CONFIG_FILE", help="config file.")
    parser.add_argument("--checkpoint_path", 
                # default='configs/conformer_wenetspeech-zh-16k_1.0/wenetspeech', 
                default='models/deepspeech2_online_wenetspeech_1.0.4/checkpoints/avg_10', 
                type=str, help="path to load checkpoint")
    parser.add_argument("--audio_file", metavar="audio_file", 
                default='zh.wav',
                help="audio file path")
    parser.add_argument("--output", type=str, 
                default='demos/speech_recognition/output/', 
                help="path of output")
    # 预测结果保存为txt
    parser.add_argument("--result_file", type=str, 
                default='', 
                help="path of save the asr result")
    args = parser.parse_args()
    print_arguments(args, globals())

    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)
    config.freeze()

    # 判断文件夹还是文件
    files = []
    refs = []
    if os.path.isdir(args.audio_file):
        files = glob.glob(args.audio_file + '*.wav')
    elif 'txt' in args.audio_file:
        with open(args.audio_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_str = line.strip('\n').split('\t')
                files.append(line_str[0])
                refs.append(line_str[1])
    else:
        files = [args.audio_file]
    num = len(files)
    logger.info('读取到 {} 个语音文件'.format(num))

    results = []

    errors_func = error_rate.char_errors
    error_rate_func = error_rate.cer

    start_time = time.time()

    # 判断模型
    if 'deepspeech2' in args.config:
        logger.info('开始预测，使用 deepspeech2 模型')
        
        t_init_s = time.time()
        exp = DeepSpeech2Tester_hub(config, args.ngpu)
        exp.setup_model()
        exp.resume(args.checkpoint_path)
        t_init_e = time.time()
        logger.info('初始化完成，用时:{}'.format(t_init_e-t_init_s))
        
        # 开始预测
        for i in range(num):
            file = files[i]

            hyp = exp.test(file)

            # 根据base64的 decode数据预测
            # 读取wav文件并编码得到一个base64 encode格式的数据
            # encode_ts = time.time()
            # with open(file, 'rb') as f:
            #     base64_bytes = base64.b64encode(f.read())
            #     audio_data = base64_bytes.decode('utf-8')
            # encode_te = time.time()
            # logger.info("b64 encode time %f s." % (encode_te - encode_ts))
            # exp.test_wav_data(audio_data)

            if refs:
                ref = refs[i]
                cer = error_rate_func(ref, hyp)
                result = '{}\t{}\t{}\t{}'.format(file, ref, hyp, cer)
                logger.info('cer: {}'.format(cer))
            else:
                result = '{}\t{}'.format(file, hyp)
            results.append(result)

    else:
        logger.info('未识别到模型')
        
    if 'txt' in args.result_file:
        with open(args.result_file,'w') as f:
            for result in results:
                f.write(result + '\n')

    use_time = time.time() - start_time
    logger.info('总用时：{}'.format(use_time))