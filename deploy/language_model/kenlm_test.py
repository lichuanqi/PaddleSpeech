import kenlm
import numpy as np
import math

# s = '淡 水 河 谷 矿 产 品 中 国 有 限 公 司'
s = '但 水 和 谷 矿 产 品 中 国 有 限 公 司'

lm_path_ad_char = 'demos/language_model/address_corpus_chars.klm'
lm_path_giga = 'configs/deepspeech2_online_wenetspeech_1.0.4/data/zh_giga.no_cna_cmn.prune01244.klm'
model=kenlm.Model(lm_path_ad_char)

# 查看N-gram
print('{0}-gram model'.format(model.order))

# model.score() 对句子进行打分
# bos=True, eos=True 给句子开头和结尾加上标记符
# 返回输入字符串的 log10 概率，得分越高，句子的组合方式越好
score = model.score(s,bos = True,eos = True)
print('score:', score)

# model.full_scores()
# score是full_scores是精简版
# full_scores会返回： (prob, ngram length, oov) 包括：概率，ngram长度，是否为oov
full_scores = model.full_scores(s)
print('full_scores: ', list(full_scores))

# 查看一句话中每个token的分数
words = ['<s>'] + list(s.replace(' ','')) + ['</s>']
for i, (prob, length, oov) in enumerate(model.full_scores(s)):
    print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i + 2 - length:i + 2])))
    if oov:
        # Find out-of-vocabulary words
        print('\t"{0}" is an OOV'.format(words[i + 1]))

# model.perplexity() 计算句子的困惑度。
perplexity =model.perplexity(s)
print('perplexity: ',perplexity)

# model.full_scores() 计算句子的困惑度
full_scores = model.full_scores(s)
prob = np.prod([math.pow(10.0, score) for score, _, _ in full_scores])
n = len(list(model.full_scores(s)))
perplexity = math.pow(prob, 1.0/n)
print(perplexity)