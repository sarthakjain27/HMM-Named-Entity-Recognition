# Named-Entity-Recognition-using-HMMs
This repository describes my approach to build my very first 1st order Hidden Markov Models from scratch for the purpose of Named Entity Recognition.


1) Learning:

learninghmm.py contains the code that takes the training tagged words file, the dictionary for converting words to integers, the label dictionary to convert labels to integers and outputs the 3 learned matrices namely Prior (initial probabilites) , Emission and the transition matrix.  

python learnhmm.py accepts 6 parameters, <input_file> <word_dictionary> <labels_dictionary> <prior_probab_output> <emission_output> <transistion_output>

2) Evaluating and Decoding:

For testing we try to approximate the P(y_t|x_1,x_2....x_T). T is the number of words in the given sentence. For this we employ Forward-Backward Algorithm by dividing the above probab in two components the forward and backward and multiplying those two to get the resultant probability. We are just considering one word prior the current word for getting the context and building our alpha (forward) and beta (backward) matrices. We are normalizing the values (column wise) in alpha beta to avoid underflow.

For prediction we are using Minimum Bayes Risk Prediction which assigns the tag with maximum probability and in case of ties, assigns the tag which appears first in the dictionary.

python forwardbackward.py <test_input_file> <index_to_word> <index_to_tag> <hmmprior> <hmmemit> <hmmtrans> <predicted_file> <metric_file>

