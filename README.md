# Persian-Farsi-Numeral-Recognition
A Jupyter-Notebook (Python) to recognize handwritten images of Persian-Farsi numerals and translate them into their English equivalent. It uses a Convolutional Neural Network trained using the dataset provided by Tarbiat Modarres University and Hoda Systems Corporation. This was done as a proof of concept for a BSCS capstone project. This project used the open source project "Hoda Dataset Reader" created by Amir Saniyan to extract the images from the original .cdb file and save them into a numpy array. 

The "Descriptive Analysis" file in the "Main" folder shows 25 examples of images from the dataset, a bargraph of the frequency of each number (6000 each), and finally a Principal Component Analysis  (PCA). A PCA is a dimensionality-reduction method that is supposed to transform a large set of variables into a smaller ones that still contains most of the information from the large set. Unfortunately my method of PCA was flawed and it showed all of the numbers grouped up together as similar, so I would need to reform the PCA to be more strict with its dimentionality reduction if I wanted the graph to be of use.

# Dataset
The NSL-KDD dataset was used for training and testing. It is widely utilized for evaluating intrusion detection systems and contains labeled network traffic data for both normal and attack scenarios.

Dataset Specifications:
Training dataset: KDDTrain+.txt
Testing dataset: KDDTest+.txt
Number of features: 41 (includes categorical and continuous variables)
Classes: Normal traffic, Denial of Service (DoS), Probe, User to Root (U2R), Remote to Local (R2L)
This project used subsets of the dataset to train and test models in customized attack scenarios:

Scenario A: DoS and U2R attacks
Scenario B: DoS and Probe attacks
Scenario C: DoS, Probe, and U2R attacks

# Instructions
  1. Run the dataExtractor.py script to create testing/training sub-datasets of the attacks based on your preference.
  2. Create a directory named "Scenarios" at the same level of the python scripts.
  3. Create subdirectories in "Scenarios", these should start with the letter "S" and will have the testing and training sub-datasets in them. For example scenario A will have a subdirectory "SA" and the files "Testing-a2-a4.csv  Training-a1-a3.csv" inside that directory. If there is no "Scenarios" directory, or the subdirectories do not start with "S", or there are not two files distinctly labelled with "Testing" and "Training" then the program will error out.
  4. Run the trainEvaluate.py script. This will train and evaluate based on the scenarios and generate the result.

# Dependencies
  Python 3.12.7\
  Tensorflow 2.17.0\
  Numpy 1.26.4\
  Keras 3.6.0\
  Matplotlib 3.9.2\
  SkLearn 1.5.2\
  Pandas 2.2.3\
   
