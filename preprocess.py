# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import re
import os
import jieba
from zhon import hanzi
from modules import Lang

# for any data type except int
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# for int data type
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

path = "./SogouC.reduced/Reduced"
test_size = 0.2
cls = ["C000008","C000010","C000013","C000014","C000016","C000020","C000022","C000023","C000024"]
max_sent_length = 50
max_sent_num = 50

def gen_data():
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    lang = Lang("Chinese") #set up a language model to keep the vocabulary
    texts = {} # Record all kinds of text
    texts_train = []
    texts_test = []    
    sentences_length = {} #Record the length of each sentence in text
    sentences_length_train = []
    sentences_length_test = []
    labels_train = []
    labels_test = []    
    for clsName in cls:
        texts[clsName] = []
        sentences_length[clsName] = []
    for clsName in cls:
        for root,dirNames,fileNames in os.walk(os.path.join(path,clsName)):
            fileNames.sort()
            for fileName in fileNames:
                with open(os.path.join(root,fileName),'rt',encoding='gbk') as file:
                    try:
                        text = file.read() #Read text(string type)
                        print("load file:%s" %os.path.join(root,fileName))
                        text = re.sub(r"(\u3000|\n|\t|\r| )","",text) #remove the blanks
                        sentences = [sent for sent in re.findall(hanzi.sentence,text) if len(sent)>=5] #punctuation
                        #Only consider a limited number of sentences to prevent memory overflow in training.
                        sentences = sentences[0:max_sent_num]
                        text = []
                        sent_len = []
                        for sent in sentences:
                            indices = []
                            count = 0
                            for token in jieba.cut(sent): #Word segmentation for each sentence.
                                #Only consider a limited number of tokens.
                                if count < max_sent_length-1: 
                                    # lang.addWord function converts token to index.
                                    indices.append(lang.addWord(token))
                                    count += 1
                                else:
                                    break
                            #Add "EOS" token as the end of the sentence.
                            indices.append(lang.word2index['EOS']) 
                            #Record the true length of each sentence in the current text
                            sent_len.append(len(indices)) 
                            while len(indices) < max_sent_length:
                                #Padding the indices to same length 
                                indices.append(lang.word2index['PAD'])
                            #Record the index sequence of this sentence
                            text.append(indices) 
                        while len(text) < max_sent_num:
                            #Padding the text to the same number of sentences
                            #to ensure that the dimensions of each text array are the same.
                            text.append([lang.word2index['PAD']]*max_sent_length)
                            sent_len.append(0)
                        texts[clsName].append(text)
                        sentences_length[clsName].append(sent_len)
                    except:
                        # Some text can not be decoded by "gbk".So we exclude these texts.
                        print("Exist encoding error in file:%s" %os.path.join(root,fileName))
    for i in range(len(cls)):
        # divide training set and test set
        x_train,x_test,y_train,y_test = train_test_split(texts[cls[i]],
                                                         sentences_length[cls[i]],
                                                         test_size = test_size,
                                                         random_state = 0
                                                         )
        texts_train += x_train
        texts_test += x_test
        sentences_length_train += y_train
        sentences_length_test += y_test
        labels_train += [i]*len(x_train)
        labels_test += [i]*len(x_test)   
    # using numpy to encapsulate data
    texts_train = np.array(texts_train,np.int32)
    texts_test = np.array(texts_test,np.int32)
    sentences_length_train = np.array(sentences_length_train,np.int32)
    sentences_length_test = np.array(sentences_length_test,np.int32)
    labels_train = np.array(labels_train,np.int64)
    labels_test = np.array(labels_test,np.int64)
    
    #write
    writer = tf.python_io.TFRecordWriter('./data/trainData.tfrecords')
    for i in range(texts_train.shape[0]):
        fea_dict = {}
        fea_dict['text'] = _bytes_feature(texts_train[i].tostring())
        fea_dict['sentences_length'] = _bytes_feature(sentences_length_train[i].tostring())
        fea_dict['label'] = _int64_feature(labels_train[i])
        features_to_write = tf.train.Example(features=tf.train.Features(feature=fea_dict))
        writer.write(features_to_write.SerializeToString())        
    writer.close()
    writer = tf.python_io.TFRecordWriter('./data/testData.tfrecords')
    for i in range(texts_test.shape[0]):
        fea_dict = {}
        fea_dict['text'] = _bytes_feature(texts_test[i].tostring())
        fea_dict['sentences_length'] = _bytes_feature(sentences_length_test[i].tostring())
        fea_dict['label'] = _int64_feature(labels_test[i])
        features_to_write = tf.train.Example(features=tf.train.Features(feature=fea_dict))
        writer.write(features_to_write.SerializeToString())        
    writer.close()    
    joblib.dump(lang,'./data/lang.pkl')
    joblib.dump(texts_train.shape[0],"./data/trainDataSize.pkl")
    joblib.dump(texts_test.shape[0],"./data/testDataSize.pkl")    
        
if __name__ == "__main__":
    gen_data()