import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


baseDir = 'Scenarios/'
scenarioDirs = list(Path(baseDir).iterdir())
scenarioDatasetDicts = {}
for dirs in scenarioDirs:
    if dirs.is_dir() and dirs.name[0] =='S':
        if dirs.name not in scenarioDatasetDicts:
            scenarioDatasetDicts[dirs.name] = {}
        scenarioDatasets = list(Path(baseDir + dirs.name).iterdir())
        if len(scenarioDatasets) == 2:
            for file in scenarioDatasets:
                if file.suffix == ".csv":
                    if "Training" in file.name:
                        scenarioDatasetDicts[dirs.name]["Training"] = baseDir + dirs.name + '/' + file.name
                    elif "Testing" in file.name:
                        scenarioDatasetDicts[dirs.name]["Testing"] = baseDir + dirs.name + '/' + file.name
                else:
                    exitString = f"{file.name} in {dirs.name} should be a csv"
                    sys.exit(exitString)
            if len(scenarioDatasetDicts[dirs.name]) != 2:
                exitString = f"{dirs.name} should have two csv files marked with Training and Testing"
                sys.exit(exitString)
        else: 
            exitString = f"There should only be 2 (training and testing) files in {dirs.name}"
            sys.exit(exitString)
    else:
        exitString = f"{dirs.name} should start with an S"
        sys.exit(exitString)

    
BatchSize=10
NumEpoch=10

for scenario in scenarioDatasetDicts:
    print(scenario)
    train_Dataset = pd.read_csv(scenarioDatasetDicts[scenario]["Training"], header=None)
    x_train = train_Dataset.iloc[:, 0:-2].values
    label_column = train_Dataset.iloc[:, -2].values
    y_train = []
    for i in range(len(label_column)):
        if label_column[i] == 'normal':
            y_train.append(0)
        else:
            y_train.append(1)
    y_train = np.array(y_train)
    
    test_Dataset = pd.read_csv(scenarioDatasetDicts[scenario]["Testing"], header=None)
    x_test = test_Dataset.iloc[:, 0:-2].values
    label_column = test_Dataset.iloc[:, -2].values
    y_test = []
    for i in range(len(label_column)):
        if label_column[i] == 'normal':
            y_test.append(0)
        else:
            y_test.append(1)
    y_test = np.array(y_test)


    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [1,2,3])],
        remainder='passthrough'                         
    )
    x_train = np.array(ct.fit_transform(x_train), dtype=float)
    x_test = np.array(ct.transform(x_test), dtype=float)
    

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)  
    x_test = sc.transform(x_test)
    
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x_train[0])))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifierHistory = classifier.fit(x_train, y_train, batch_size = BatchSize, epochs = NumEpoch)

    loss, accuracy = classifier.evaluate(x_train, y_train)
    print('Print the loss and the accuracy of the model on the dataset')
    print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))

    y_pred = classifier.predict(x_test)
    y_pred = (y_pred > 0.9)   

    cm = confusion_matrix(y_test, y_pred)
    print('Print the Confusion Matrix:')
    print('[ TN, FP ]')
    print('[ FN, TP ]=')
    print(cm)


    print('Plot the accuracy')
    plt.plot(classifierHistory.history['accuracy'])
    title = scenario + ' model accuracy'
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('accuracy_sample.png')
    plt.show()

    print('Plot the loss')
    plt.plot(classifierHistory.history['loss'])
    title = scenario + ' model loss'
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('loss_sample.png')
    plt.show()
    
    print("Finished a scenario")
    
    
    