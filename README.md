# Network Anomaly Detection
A project to detect network anomalies using machine learning techniques, specifically Feed-Forward Neural Networks (FNN). This project employs the NSL-KDD dataset to identify abnormal patterns in network traffic, indicating potential threats such as intrusions or malware. The project focuses on data preprocessing, training customized models, and evaluating their performance under various attack scenarios. It was developed as part of an advanced cybersecurity coursework project.


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
     ![image](https://github.com/user-attachments/assets/9d2849da-21b1-4138-9cd2-831bb103154f)
  4. Run the trainEvaluate.py script. This will train and evaluate based on the scenarios and generate the result. I use Spyder for its nice to use interface for viewing plots.
![image](https://github.com/user-attachments/assets/77767145-45eb-4d3b-adb5-6ecc6fce7373)
# Dependencies
  Python 3.12.7\
  Tensorflow 2.17.0\
  Numpy 1.26.4\
  Keras 3.6.0\
  Matplotlib 3.9.2\
  SkLearn 1.5.2\
  Pandas 2.2.3\
   
