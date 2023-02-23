# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import sys
import numpy as np
import librosa
import time
import base64
import warnings
from loguru import logger

from pathlib import Path
from typing import Union
import uvicorn
import soundfile
from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware
from yacs.config import CfgNode
import paddle

# 模型相关
from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.utility import UpdateConfig

# 部署相关
from paddlespeech.server.restful.request import ASRRequest
from paddlespeech.server.restful.response import ASRResponse
from paddlespeech.server.restful.response import ErrorResponse
from paddlespeech.server.utils.config import get_config


class Log():
    """loguru的封装

    Params
        level : 日志等级
        stdout: 是否输出日志到控制台
        logdir: 日志保存路径
        level : 日期等级
    """
    def __init__(self,
                 level='DEBUG',
                 stdout=True, 
                 logdir=False) -> None:
        self.logger = logger
        self.logger.remove()

        if stdout:
            # 日志输出到控制台
            self.logger.add(sys.stdout)

        if logdir:
            # 保存日志
            logname = os.path.join(logdir, '{time}.log')
            self.logger.add(logname, level=level,
                rotation="100MB", encoding="utf-8",
                enqueue=True, retention="100 days")

    def get_logger(self):
        return self.logger


class DeepSpeech2Tester_hub():
    """DeepSpeech2核心函数"""
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


    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def run_wav_data(self, audio_data):
        self.model.eval()
        self.audio_data = audio_data

        # 解码
        decode_ts = time.time()
        audio_decode = base64.b64decode(self.audio_data)
        decode_te = time.time()
        logger.debug("base64 decode time %f s." % (decode_te - decode_ts))

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
        logger.debug(f"audio shape: {audio.shape}")

        # fbank
        feat = self.preprocessing(audio, **self.preprocess_args)
        logger.debug(f"feat shape: {feat.shape}")

        audio_len = paddle.to_tensor(feat.shape[0])
        audio = paddle.to_tensor(feat, dtype='float32').unsqueeze(axis=0)

        result_transcripts = self.compute_result_transcripts(
            audio, audio_len, self.text_feature.vocab_list, self.config.decode)
        logger.info("识别结果: " + result_transcripts[0])

        return result_transcripts[0]

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


warnings.filterwarnings("ignore")
logger = Log(logdir='deploy/logs').get_logger()

app = FastAPI(
    title="PaddleSpeech Serving API", 
    description="Api", 
    version="v0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# change yaml file here
config_file = "deploy/speech_server/conf/asr_ds2.yaml"
server_config = get_config(config_file)

# 加载模型
t_init_s = time.time()
cfg_path = server_config.asr_python.cfg_path
ckpt_path = server_config.asr_python.ckpt_path

config = CfgNode(new_allowed=True)
config.merge_from_file(cfg_path)
config.freeze()

if 'ds2' in config_file:
    logger.info('初始化，使用 deepspeech2 模型')
    exp = DeepSpeech2Tester_hub(config, ngpu=1)
    exp.setup_model()
    exp.resume(ckpt_path)
else:
    logger.info('未检测到模型，初始化失败')
    sys.exit()

t_init_e = time.time()
logger.info('初始化完成，用时:{}'.format(t_init_e-t_init_s))

@app.get('/')
def root():
    """
    根目录
    """
    return {'success': True,
            'message': 'Hello PaddleSpeech'}

@app.post("/paddlespeech/asr", response_model=Union[ASRResponse, ErrorResponse])
def asr(request_body: ASRRequest):
    """asr api 

    Args:
        request_body (ASRRequest): [description]

    Returns:
        json: [description]
    """
    rsl = exp.run_wav_data(request_body.audio)

    response = {
        "success": True,
        "code": 200,
        "message": {
            "description": "success"
        },
        "result": {
            "transcription": rsl
        }
    }

    return response


if __name__ == "__main__":

    uvicorn.run(
        "server_start_asr_ds2:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers)