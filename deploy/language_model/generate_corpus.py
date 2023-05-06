"""
根据句子生成kenlm训练用的词料库

输入: text.txt
        我爱中国
        ...

输出: corpus_chars.txt
        (字粒度)我 爱 中 国
                ...
        (词粒度)我 爱 中国
                ...
"""
import sys
from pathlib import Path
import string
import jieba


def read_txt(txt_path):
    '''
    读取txt文件中的文字,删除所有标点符号
    txt_path: txt文件路径
    return: 
    '''
    txt_clean = []

    with open(txt_path,'r') as f:
        lines = f.readlines()
    
    for line in lines:
        strs_clean = clean_str(line.strip())
        strs_str = num2str(strs_clean)
        txt_clean.append(strs_str)

    return txt_clean


def clean_str(strs:str) -> str:
    """
    去除所有的标点符号等特殊字符
    中国（科技）出版传媒股份有限公司#￥%…… -> 中国科技出版传媒股份有限公司
    """
    strs = str(strs)
    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    punctuation = r"""!"#￥$%&'（）《》()*+,-./:;<=>?@[\]^_`{|}"""
    chin = ascii_lowercase + ascii_uppercase + punctuation
    table = str.maketrans('', '', chin)
    strs_clean = strs.translate(table)

    return strs_clean


def num2str(strs:str) -> str:
    """
    将字符串中的数字转换为汉字
    解放军301医院 -> 解放军三零一医院
    """
    strs=str(strs)
    strs_new=""
    num_dict={"0":u"零","1":u"一","2":u"二","3":u"三","4":u"四","5":u"五","6":u"六","7":u"七","8":u"八","9":u"九"}
    listnum=list(strs)
    # print(listnum)
    shu=[]
    for i in listnum:
        if i in num_dict:
            shu.append(num_dict[i])
        else:
            shu.append(i)
    strs_new="".join(shu)

    return strs_new
    

def cut_chars(strings):
    """
    中国科技出版传媒股份有限公司 -> 中 国 科 技 出 版 传 媒 股 份 有 限 公 司
    """

    strings_new = ' '.join(strings)

    return strings_new


def cut_words(strs:str):
    """
    使用jieba对字符串进行切词
    中国科技出版传媒股份有限公司 -> 中国 科技 出版 传媒 股份 有限 公司
    """
    cut = list(jieba.cut(strs))
    strings_new = ' '.join(cut)
    return strings_new


def write_txt(txts, savepath, mode='char'):
    '''
    将结果保存为txt文件
    txt: ['中 国 科 技 出 版 传 媒 股 份 有 限 公 司', '中 国 石 化 国 际 事 业 有 限 公 司 北 京 招 标 中 心']
    save_path: 保存地址
    '''
    savefile = Path(savepath).joinpath('address_corpus_%s.txt'%mode)
    print('savefile: %s'%savefile)

    with open(savefile, 'w', encoding='UTF-8') as f:
        for txts in txts:
            # 字粒度
            if mode == 'char':
                txts_cut = cut_chars(txts)

            # 词粒度
            elif mode == 'word':
                txts_cut = cut_words(txts)

            else:
                break

            f.write(txts_cut+'\n')

    print('txt save finished')


def fun_test():

    print(cut_chars('中国科技出版传媒股份有限公司'))
    print(clean_str('中国（科技）出版传媒股份有限公司#￥%……'))
    print(num2str('解放军301医院'))
    print(cut_words('中国科技出版传媒股份有限公司'))
    print('测试通过')


if __name__ == '__main__':

    fun_test()

    text_path = 'deploy/language_model/20230506_jiyao/train_ngram.txt'
    savepath = 'deploy/language_model/20230506_jiyao/'

    txts = read_txt(text_path)
    write_txt(txts=txts, savepath=savepath, mode='char')