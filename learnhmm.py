from __future__ import division
import sys
__author__ = 'SARTHAK JAIN'


def create_data_matrix(file,word_dict,label_dict):

    with open(file,'r') as f:
        no_split_data=[line.split() for line in f]
        one_line_raw=[j for i in no_split_data for j in i]


    with open(file, 'r') as f:
        data=[line.replace("_"," ").split() for line in f]
    
    raw_data=[]
    conv_data=[]
    raw_label=[]
    conv_label=[]
    one_line_label_trans=[]
    for each in data:
        word=each[::2]
        lab=each[1::2]
        for i in range(0,len(lab)-1):
            one_line_label_trans.append(lab[i]+'_'+lab[i+1])
        raw_data.append(word)
        raw_label.append(lab)
        conv_data.append([word_dict[k] for k in word])
        conv_label.append([label_dict[k] for k in lab])
    return one_line_raw,one_line_label_trans,raw_data,conv_data,raw_label,conv_label

def create_dict(file):
    dict={}
    with open(file,'r') as f:
        for line_no, word in enumerate(f):
            dict[word.rstrip('\n')]=line_no

    return dict

def create_prior(file,train_raw_label,label_dict):
    probab_value=[0]*len(label_dict)

    first_col=[i[0] for i in train_raw_label]
    label_count={label_dict[x]:first_col.count(x)+1 for x in label_dict}
    normalizing_factor=sum(label_count[x] for x in label_count)

    for keys in label_count:
        probab_value[keys]=label_count[keys]/normalizing_factor

    with open(file, 'w') as f:
        for item in probab_value:
            f.write("%s\n" % item)

def create_emmit(file,train_conv_data,train_conv_label,word_dict,label_dict):
    x = [[1]*len(word_dict) for i in range(len(label_dict))]
    for row,each_sentence in enumerate(train_conv_data):
        for col,word_in_sent in enumerate(each_sentence):
            x[train_conv_label[row][col]][word_in_sent]+=1

    for row,each_row in enumerate(x):
        normalizing_factor=sum(each_row)
        for col,each in enumerate(each_row):
            x[row][col]=str(x[row][col]/normalizing_factor)+' '
        x[row][col]=x[row][col].rstrip(' ')

    with open(file,'w') as f:
        for each_row in x:
            for each_col in each_row:
                f.write(each_col)
            f.write('\n')

    return x

def create_trans(file,lab_trans,label_dict):
    x = [[1]*len(label_dict) for i in range(len(label_dict))]
    for each_row in label_dict:
        for each_col in label_dict:
            word=each_row+"_"+each_col
            x[label_dict[each_row]][label_dict[each_col]]+=lab_trans.count(word)

    for row,each_row in enumerate(x):
        normalizing_factor=sum(each_row)
        for col,each in enumerate(each_row):
            x[row][col]=str(x[row][col]/normalizing_factor)+' '
        x[row][col]=x[row][col].rstrip(' ')

    with open(file,'w') as f:
        for each_row in x:
            for each_col in each_row:
                f.write(each_col)
            f.write('\n')

    return x

if __name__ == "__main__":
    word_dict=create_dict(sys.argv[2])
    label_dict=create_dict(sys.argv[3])

    one_line_data_raw,one_line_lab_trans_raw,train_raw_data, train_conv_data, train_raw_label,train_conv_label=create_data_matrix(sys.argv[1],word_dict,label_dict)

    create_prior(sys.argv[4],train_raw_label,label_dict)
    create_emmit(sys.argv[5],train_conv_data,train_conv_label,word_dict,label_dict)
    create_trans(sys.argv[6],one_line_lab_trans_raw,label_dict)
