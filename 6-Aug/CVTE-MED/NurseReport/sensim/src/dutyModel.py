# coding:utf-8  
'''
Created on 2019年7月2日

@author: cvter
'''
import argparse
import jieba
from gensim.models import doc2vec
import csv

def parse_args():#define the paramter of program
    parser = argparse.ArgumentParser(description="dutyModel.")
    parser.add_argument('--fileName', nargs='?', default='med.csv',help='fileName.')
    return parser.parse_args(args=[])

def get_stop_words():#load the stopwords 
    spath = '../data/stopword.txt'
    stopwords = [line.strip() for line in open(spath, 'r', encoding='GBK').readlines()]  
    return stopwords

def get_lineText(textpath): #get the data and tokenize
    rows = csv.reader(open(textpath,'r',encoding='utf-8'))
    lineText = []
    rawText = []
    stopwords = get_stop_words()
    for r in rows:
        rawText.append(r[0])
        seg_list = jieba.lcut(r[0].strip()) 
        txt_list = [' '.join(seg) for seg in seg_list if seg not in stopwords]
        lineText.append(txt_list)
    return lineText,rawText

def train_doc2vec_model(tagged_data):
    max_epochs = 100
    vec_size = 20
    alpha = 0.025
    #If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). 
    model = doc2vec.Doc2Vec(size=vec_size,alpha=alpha, min_alpha=0.00025,min_count=1,dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    
    model.save("../data/d2v.model")
    print("Doc2Vec Model Saved")

if __name__ == '__main__':
    #1.读取参数
    args = parse_args()
    fileName = args.fileName
    #2.加载文本，utf-8格式
    lineText,rawText = get_lineText('../data/'+fileName)
    print ("The texts %d has been loaded successfully in the file %s" % (len(lineText),fileName))
    with open('../data/raw.txt','w', encoding='utf-8') as fw:
        lists=[line+"\n" for line in rawText]
        fw.writelines(lists)
    #3.词典生成和训练模型
    tagged_data = [doc2vec.TaggedDocument(words=line, tags=[str(i)]) for i, line in enumerate(lineText)]
    train_doc2vec_model(tagged_data)
    
