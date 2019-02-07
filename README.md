# QuestionClassifier
The project deals with creating a classifier that can identify whether or not a provided series of words is a question .

Preprocessing for input-text given for classifier: Includes finding minimum size, maximum size, mean size, standard deviation                         among given sntences for training the model with accordingly sized dataset. 

Dataset choosen for Training and Testing The Model: SQuAD 1.0 Dataset
  Pre-processing steps: Read the SQuAD dataset Train, Validation and Test json files and 
                        capture questions and non-questions (context).
                        Removing Question marks from half of the questions, and periods from half of the context texts.
                        Removing large size context paragraphs and limiting them to a size less than 60 words.
                        Storing the processed Train, Validation and Test files as csv with labels question or not-question.

Model Deisgn: Includes an end-to-end fully-connected network that uses a BiLSTM network connected with a linear layer, trained                on pre-processed SQuAD dataset, to classify a given sentence as a question or not-question, using PyTorch.

Classification: The trained BiLSTM model on SQuAD dataset is loaded and used for claasifying a sentence as question or not-                   question in an output file.
                        
                        
  
