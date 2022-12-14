# 语音识别客户端


## 程序打包

```bash
# 开发环境生成依赖库列表
cd /d D:\CODE\PaddleSpeech\deploy\speech_client_pyqt
conda activate paddle
pipreqs ./ --encoding=utf8

# 进入打包化境安装最小依赖
conda activate app
pip install -r requirements.txt 
pip install pyinstaller

# 打包
# 产生单个的可执行文件
pyinstaller -F main.py -w -n 语音录入

# 产生一个目录（包含多个文件）作为可执行程序
pyinstaller -D main.py
```