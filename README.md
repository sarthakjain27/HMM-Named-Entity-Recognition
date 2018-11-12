# Named-Entity-Recognition-using-HMMs
This repository describes my approach to build my very first 1st order Hidden Markov Models from scratch for the purpose of Named Entity Recognition.


1) Learning:

learninghmm.py contains the code that takes the training tagged words file, the dictionary for converting words to integers, the label dictionary to convert labels to integers and outputs the 3 learned matrices namely Prior (initial probabilites) , Emission and the transition matrix. 

learnhmm.py accepts 6 parameters, <input_file> <word_dictionary> <labels_dictionary> <prior_probab_output> <emission_output> <transistion_output>
