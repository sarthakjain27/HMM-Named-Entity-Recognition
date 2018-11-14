from __future__ import division
import sys
import numpy as np
import math
__author__ = 'SARTHAK JAIN'

def read_matrices():
    with open(sys.argv[4],'r') as f:
        pi=[map(float,line.split()) for line in f]

    pi=np.asarray(pi)

    with open(sys.argv[5],'r') as f:
        b=[map(float,line.split()) for line in f]

    b=np.asarray(b)

    with open(sys.argv[6],'r') as f:
        a=[map(float,line.split()) for line in f]

    a=np.asarray(a)

    return pi,a,b

def create_dict(file):
    dict={}
    with open(file,'r') as f:
        for line_no, word in enumerate(f):
            dict[word.rstrip('\n')]=line_no

    return dict

def create_test_data_matrix(word_dict,label_dict):
    with open(sys.argv[1],'r') as f:
        no_split_data=[line.split() for line in f]
        one_line_raw=[j for i in no_split_data for j in i]

    with open(sys.argv[1], 'r') as f:
        data=[line.replace("_"," ").split() for line in f]

    raw_data=[]
    conv_data=[]
    raw_label=[]
    conv_label=[]
    for each in data:
        word=each[::2]
        lab=each[1::2]
        raw_data.append(word)
        raw_label.append(lab)
        conv_data.append([word_dict[k] for k in word])
        conv_label.append([label_dict[k] for k in lab])

    return one_line_raw,raw_data,conv_data,raw_label,conv_label

def forward(conv_sentence,prior,a,b):
    alpha=np.zeros((len(prior),len(conv_sentence)))
    each=np.multiply(prior,b[:,conv_sentence[0]].reshape((-1,1))).flatten()
    alpha[:,0]=each/each.sum()
    a_trans=np.transpose(a)
    for word_index in range(1,len(conv_sentence)):
        each=np.multiply(b[:,conv_sentence[word_index]],(np.dot(a_trans,alpha[:,word_index-1]))).flatten()
        alpha[:,word_index]=each/each.sum()
    alpha[:,len(conv_sentence)-1]=each
    return alpha

def backward(conv_sentence,prior,a,b):
    num_words=len(conv_sentence)
    num_labels=len(prior)
    beta=np.ones((num_labels,num_words))
    beta[:,num_words-1]=beta[:,num_words-1]/beta[:,num_words-1].sum()
    for word_index in range(len(conv_sentence)-1,0,-1):
        each=np.dot(a,np.multiply(b[:,conv_sentence[word_index]].reshape((-1,1)),beta[:,word_index].reshape((-1,1)))).flatten()
        beta[:,word_index-1]=each/each.sum()
    return beta

def accuracy(flatten_original,pred):

    flatten_predict = [item for sublist in pred for item in sublist]
    pairwise = zip (flatten_original, flatten_predict)
    matched_digits = [idx for idx, pair in enumerate(pairwise) if pair[0] == pair[1]]

    return (float(len(matched_digits))/len(flatten_original))

def predict(one_line_raw_data_no_split,raw_test_data,conv_test_data,label_dict,pi,a,b):
    x = [['']*len(raw_test_data[i]) for i in range(len(raw_test_data))]

    alpha_T_sum=0.0

    for index,each_sentence in enumerate(raw_test_data):
        alpha=forward(conv_test_data[index],pi,a,b)
        alpha_T_sum+=math.log(np.sum(alpha[:,len(each_sentence)-1]))

        beta=backward(conv_test_data[index],pi,a,b)

        alpha_dot_beta=np.multiply(alpha,beta)
        max_label_index=alpha_dot_beta.argmax(axis=0)
        each_sent_predict_label=[label_dict.keys()[label_dict.values().index(k)] for k in max_label_index]
        x[index]=[each_sentence[k]+'_'+each_sent_predict_label[k] for k in range(len(each_sentence))]

    with open(sys.argv[7],'w') as f:
        for each_row in x:
            for each_col in each_row:
                f.write(each_col+' ')
            f.write('\n')

    accur=accuracy(one_line_raw_data_no_split,x)
    all=alpha_T_sum/len(raw_test_data)

    with open(sys.argv[8],'w') as f:
        f.write('Average Log-Likelihood: {}'.format(all))
        f.write('\n')
        f.write('Accuracy: {}'.format(accur))
        f.write('\n')

    return x

if __name__ == "__main__":
    word_dict=create_dict(sys.argv[2])
    label_dict=create_dict(sys.argv[3])

    pi,a,b=read_matrices()

    one_line_raw_data_no_split,raw_test_data,conv_test_data,raw_test_label,conv_test_label=create_test_data_matrix(word_dict,label_dict)

    predict(one_line_raw_data_no_split,raw_test_data,conv_test_data,label_dict,pi,a,b)

