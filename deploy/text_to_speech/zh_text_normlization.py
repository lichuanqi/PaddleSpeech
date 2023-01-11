
from paddlespeech.t2s.frontend.zh_normalization.text_normlization import \
    TextNormalizer

# 预处理 1 文本标准化
txt = '（中国）邮政001'
TN = TextNormalizer()
txt_tn = TN.normalize(txt)[0]
print(txt_tn)