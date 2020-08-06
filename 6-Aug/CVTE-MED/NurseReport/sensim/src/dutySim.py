# coding:utf-8  
'''
Created on 2019年7月2日

@author: cvter
'''
import argparse
import jieba
from gensim.models import doc2vec

def parse_args():#define the paramter of program
    parser = argparse.ArgumentParser(description="dutySim.")
    parser.add_argument('--text', nargs='?', default='患者神志清 , 精神差 , 心电监测示:窦性心率 , 律齐。',help='text.')
    parser.add_argument('--topK', nargs='?', default='10',help='topK.')
    return parser.parse_args(args=[])

def get_stop_words():#load the stopwords 
    spath = '../data/stopword.txt'
    stopwords = [line.strip() for line in open(spath, 'r', encoding='GBK').readlines()]  
    return stopwords

def get_lineText(text): #get the data and tokenize
    stopwords = get_stop_words()
    seg_list = jieba.lcut(text.strip()) 
    lineText = [' '.join(seg) for seg in seg_list if seg not in stopwords]
    return lineText

def get_most_similar(lineText,topK):
    model= doc2vec.Doc2Vec.load("../data/d2v.model")
    predVec = model.infer_vector(lineText)
    '''
    mostSim=model.docvecs.most_similar(0)
    sims = model.docvecs.similarity(1,2)#计算两两相似度
    docvec =model.docvecs[1]#返回对应的向量
    Returns:    Sequence of (doctag/index, similarity).
    Return type:    list of ({str, int}, float)
    '''
    mostSim=model.docvecs.most_similar([predVec], topn=int(topK))

    return mostSim

if __name__ == '__main__':
    #1.读取参数
    args = parse_args()
    text = args.text
    topK = args.topK
    #2.分词
    lineText = get_lineText(text)
    #3.匹配相似度最高k项
    mostSim = get_most_similar(lineText,topK)
    rawText = [line for line in open('../data/raw.txt', 'r',encoding='utf8').readlines()]
    simItems = []
    for i, sim in mostSim:
        strtxt = rawText[int(i)]
        simItems.append([i,sim,strtxt])
    with open('../data/sim.txt','w',encoding='utf-8') as fw: #返回最高相似度结果
        lists=[str(line)+'\n' for line in simItems]
        fw.writelines(lists)
    
    
