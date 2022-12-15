# 语音识别客户端


## 程序打包

```bash
# 开发环境生成依赖库列表
cd /d D:\CODE\PaddleSpeech\deploy\speech_client_pyqt
conda activate paddle
pipreqs ./ --encoding=utf8

# 进入打包化境安装最小依赖
cd /d D:\CODE\PaddleSpeech\deploy\speech_client_pyqt
conda activate app
pip install -r requirements.txt 
pip install pyinstaller

# 打包
# 产生单个的可执行文件
pyinstaller -F main.py -w -n 语音录入

# 产生一个目录（包含多个文件）作为可执行程序
pyinstaller -D main.py
```

## 使用说明

说明
1. 程序启动时会在当前目录下新建`files`文件夹,并根据程序启动日期保存录音文件和识别结果


## 版本更新

`开发中`
- [ ] 程序初始化过程中检测api接口是否能正常使用
- [ ] 程序关闭时将表格中的数据保存到缓存路径下,避免数据丢失

`v0.2`

- 使用pyqt重构程序界面
- 新增表格内容的导入和导出