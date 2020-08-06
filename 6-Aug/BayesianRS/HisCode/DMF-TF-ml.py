# -*- Encoding:UTF-8 -*-
'''
@author: Jason.F
@data: 2019.07.16
@function: Implementing DMF with Torch  
           Dataset: Movielen Dataset(ml-1m) 
           Evaluating: hitradio,ndcg
           https://www.ijcai.org/proceedings/2017/0447.pdf
           https://github.com/RuidongZ/Deep_Matrix_Factorization_Models
'''
import tensorflow as tf
import numpy as np
import argparse
import os
import heapq
import math
import sys

#define class DMF
class DMF():
    
    def getTrainMatrix(self):
        train_matrix = np.zeros([self.maxu, self.maxi], dtype=np.float32)
        for i in self.trainset:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)
    
    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.getTrainMatrix())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        
    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)
        
    def add_model(self):#network structure
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.maxi, self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.maxu, self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)
        
    def add_loss(self):#normalized cross-entropy loss
        regRate = self.rate / self.maxr
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss
    
    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)
        
    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        '''
        self.saver = tf.train.Saver()
        if os.path.exists("./checkPoint/"):
            [os.remove(f) for f in os.listdir("./checkPoint/")]
        else:
            os.mkdir("./checkPoint/")
        '''
        
    def __init__(self, trainset, testset, maxr, maxu, maxi, K=64, neg_num=4, lr=0.001):
        self.trainset = trainset
        self.testset = testset
        self.maxr = maxr
        self.maxu = maxu+1
        self.maxi = maxi+1
        self.neg_num = neg_num
        self.lr = lr
        self.batchSize = 256
        #init model
        self.add_embedding_matrix()
        self.add_placeholders()
        self.userLayer = [512, K]#user hidden layers
        self.itemLayer = [1024, K]#item hidden layers
        self.add_model()
        self.add_loss()
        self.add_train_step()
        self.init_sess()
        
        #handle data
        self.traindict = self.getTrainDict()
        self.train_u, self.train_i, self.train_r = self.getInstances()
        self.testPosNeg = self.getTestPosNeg()
               
    def getTrainDict(self):
        dataDict = {}
        for i in self.trainset:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict     
       
    def getInstances(self):
        user = []
        item = []
        rate = []
        for i in self.trainset:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(self.neg_num):
                j = np.random.randint(self.maxi)
                while (i[0], j) in self.traindict:
                    j = np.random.randint(self.maxi)
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)
    
    def getTestPosNeg(self):
        user = []
        item = []
        u_prev = testset[0][0]
        tmp_user = []
        tmp_item = []
        for u, i in testset:
            if u_prev ==u:
                tmp_user.append(u)
                tmp_item.append(i)
            else:
                user.append(tmp_user)
                item.append(tmp_item)
                tmp_user = []
                tmp_item = []
                tmp_user.append(u)
                tmp_item.append(i)
            u_prev = u
        return [np.array(user), np.array(item)]
    
    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}
    
    def run_epoch(self, verbose=10):
        train_len = len(self.train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = self.train_u[shuffled_idx]
        train_i = self.train_i[shuffled_idx]
        train_r = self.train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            
            _, tmp_loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            
            losses.append(tmp_loss)
            #if verbose and i % verbose == 0:
            #    sys.stdout.write('\r{} / {} : loss = {}'.format(i, num_batches, np.mean(losses[-verbose:])))
            #    sys.stdout.flush()
        loss = np.mean(losses)
        return loss
    
    def evaluate(self, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0
        
        hr =[]
        NDCG = []
        testUser = self.testPosNeg[0]
        testItem = self.testPosNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = self.sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)
    
#1.loading dataset
def getTrainset(filePath):
    trainset = []
    maxu = 0 
    maxi = 0 
    maxr = 0.0
    with open(filePath, 'r') as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i, rating = int(arr[0]), int(arr[1]), float(arr[2])
            trainset.append([int(arr[0]), int(arr[1]), float(arr[2])])
            if rating > maxr: maxr = rating
            if u > maxu: maxu = u
            if i > maxi: maxi = i
            line = fd.readline()
        return trainset, maxr, maxu, maxi

def getTestset(filePath):
    testset = []
    with open(filePath, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            testset.append([u, eval(arr[0])[1]])#one postive item
            for i in arr[1:]:
                testset.append([u, int(i)]) #99 negative items
            line = fd.readline()
    return testset     
    
if __name__ == '__main__':
    trainset, maxr, maxu, maxi = getTrainset("/data/fjsdata/ctKngBase/ml/ml-1m.train.rating")
    testset = getTestset("/data/fjsdata/ctKngBase/ml/ml-1m.test.negative")
    print('Dataset Statistics: Interaction = %d, User = %d, Item = %d, Sparsity = %.4f' % (len(trainset), maxu+1, maxi+1, len(trainset)/(maxu*maxr)))
    print ("%3s%20s%20s" % ('K','HR@10', 'NDCG@10'))
    for K in [8,16,32,64]:
        mdl = DMF(trainset, testset, maxr, maxu, maxi, K=K, neg_num = 4, lr = 0.001) #the ratio of samples is pos 1: neg 4
        best_hr, best_ndcg= 0.0, 0.0
        for epoch in range(1):#epoch=20
            loss = mdl.run_epoch()
            print("\nMean loss in this epoch is: {}".format(loss))
            hr, ndcg = mdl.evaluate(topK=10)#default recommend top 10 items
            if hr>best_hr: best_hr=hr
            if ndcg>best_ndcg: best_ndcg=ndcg
        print ("%3d%20.6f%20.6f" % (K, best_hr, best_ndcg))
        
'''
nohup python -u DMF-TF-ml.py > dmftf-ml.log  &


'''