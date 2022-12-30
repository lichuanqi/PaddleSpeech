# 语音识别客户端

## 使用流程及说明

使用流程

1. 【输条码】 在最上方手动输入或者通过扫描枪扫前输入条码
2. 【录声音】 点击 `开始` 按钮开始录音, 收件地址说完后点击 `结束` 按钮
   表格区域就会出现刚刚输入的条码和录音文件的保存路径
3. 【转文字】 点击 `预测` 按钮调用语音识别接口
   预测完成后会在表格中出现对应的文字结果
4. 【导结果】 点击 `导出` 按钮将表格中的结果导出至txt文件

使用说明

1. 程序启动时会在当前目录下新建`files`文件夹,并根据程序启动日期保存录音文件和识别结果


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
# 生成单个的可执行文件
pyinstaller -F main.py -w -n 语音录入

# 生成包含多个文件的目录作为可执行程序
pyinstaller -D main.py
```


## 项目结构

```python
|—— main.py   # 程序主入口
|—— api.py    # 语音识别接口相关函数
|—— utils.py  # 公共函数
|—— test.py   # 函数测试
|—— files     # 
|   |—— 2022-12-30
|   |   |—— data_list.txt
|   |   |—— 10001.wav
```


## 版本更新

`开发中`
- [x] 程序初始化过程中检测api接口是否能正常使用
- [ ] 程序关闭时将表格中的数据保存到缓存路径下,避免数据丢失

`v0.2`
- 使用pyqt重构程序界面
- 新增表格内容的导入和导出