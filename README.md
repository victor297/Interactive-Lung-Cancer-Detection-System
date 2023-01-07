# Lung Cancer Detection
In this project, I used multiple python libraries to detect whether a person is suffering from lung cancer. I collected training data from [Kaggle](https://www.kaggle.com/).
## Installation of required tools

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas, sklearn.
```python
pip install numpy
pip install pandas
pip install sklearn
```
## Importing required modules

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
```

## Read data from file
In [pandas](https://pandas.pydata.org/), we have a function named read_csv, which is used to read data from csv files. After retrieving data from file, we store it in a variable 'data'.
```python
data = pd.read_csv('D:/MyProjects/Lung_Cancer_Detection/cancer_data.csv') #This data is downloded from kaggle
```
## Conversion of data into numeric values
Labelling data for better understanding.
```python
data.loc[data['LUNG_CANCER'] == 'YES', 'label', ] = 1
data.loc[data['LUNG_CANCER'] == 'NO', 'label', ] = 0
data.loc[data['GENDER'] == 'M', 'GENDER', ] = 1
data.loc[data['GENDER'] == 'F', 'GENDER', ] = 0
```
## Differentiation of input data and output label

```python
X = data.drop(['LUNG_CANCER', 'label'], axis=1)
Y = data['label'].astype(int)
```
## Differentiation of training and testing data
X_train, X_test, Y_train, Y_test are four matrices which store respective data. 'test_size' represents percentage of data that should be used for testing. Here 'test_size = 0.3' represents that 30% of data will be used for testing and rest 70% will be used for training data. Also, in the data 2 represents 'YES' and 1 represents 'NO'.\
Optionally, we can use random_state variable to have a control on the way of how data splits.\
**Note : train_test_split is an inbuilt function from sklearn.linear_model**

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
```
# Implementation of Models
Since we are dealing with a Classification problem, we use many classification techniques. I used the following models : 
* Logistic Regression
* Decision Tree
* Support Vector Machine
* K - Nearest Neighbours
* Random Forest Classifier
* Naive Bayes

## Logistic Regression
By default in LBFGS, the maximum itertaions are 100. If there is a need for increase in iterations, we can pass a parameter 'max_iter' as shown below.
```python
Logistic_Regression_model = LogisticRegression(max_iter=1000)
Logistic_Regression_model.fit(X_train.values, Y_train.values)
```
## Decision Tree
```python
Decision_Tree_model = DecisionTreeClassifier()
Decision_Tree_model.fit(X_train.values, Y_train.values)
```
### For plotting tree
```python
plt.figure(figsize=(20,10))
plot_tree(decision_tree = Decision_Tree_model, filled=True, feature_names=X.columns, rounded = True)
plt.show()
```
## Support Vector Machine
```python
SVM_model = SVC()
SVM_model.fit(X_train.values, Y_train.values)
```
## K - Nearest Neighbours Model
By default, nearest neighbours is 5
```python 
KNN_model = KNeighborsClassifier(4) 
KNN_model.fit(X_train.values, Y_train.values)
```
## Random Forest Classifier
```python
Random_Forest_Classifier = RandomForestClassifier()
Random_Forest_Classifier.fit(X_train.values, Y_train.values)
```
## Naive Bayes Model
```python
Naive_Bayes_model = GaussianNB()
Naive_Bayes_model.fit(X_train.values, Y_train.values)
```


## Prediction and accuracy
From sklearn.metrics, we have already imported accuracy_score, which will be used here to calculate how accurately our model is able to predict data.\
At first we calculate the predicted data and store it in variable ptrain, then compare it with original data (Y_train).
```python
ptrain = Model_Name.predict(X_train.values)
accuracy = accuracy_score(Y_train, ptrain)
print(accuracy)
```

The above task is performed again on test data, to check for overfitting.
```python
ptest = Model_Name.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)
```

# Convert function 
This function is used to convert string to corresponding integer.\
To better understand this function kindly refer main function.
```python
def convert(str, l):
    if (str == 'Yes'):
        l.append(2)
    else:
        l.append(1)
```
## The main function
In the main function, we take input for every value from user, then convert that respond to an integer using convert function.
```python
def main():
    user_inputs = []
    gender = input("Please enter your gender (M/F) : ")
    if (gender == 'F'):
        user_inputs.append(0)
    else:
        user_inputs.append(1)

    age = input("Please enter your age : ")
    user_inputs.append(age)
    smoke = input("Do you smoke? (Yes/No) : ")
    func(smoke, user_inputs)
    yellow_fingers = input("Do you have yellow fingers? (Yes/No): ")
    func(yellow_fingers, user_inputs)
    anxiety = input("Do you have anxiety issues? (Yes/No) : ")
    func(anxiety, user_inputs)
    chronic = input("Do you have any chronic diseases? (Yes/No) : ")
    func(chronic, user_inputs)
    fatigue = input("Do you experience fatigue? (Yes/No) : ")
    func(fatigue, user_inputs)
    allergy = input("Do you have any allergies? (Yes/No) : ")
    func(allergy, user_inputs)
    wheezing = input("Do you wheeze? (Yes/No) : ")
    func(wheezing, user_inputs)
    alcohol = input("Do you consume alcohol? (Yes/No) : ")
    func(alcohol, user_inputs)
    cough = input("Do you cough on a regular basis? (Yes/No) : ")
    func(cough, user_inputs)
    breathing = input("Do you have any kind of difficulity in breathing? (Yes/No) : ")
    func(breathing, user_inputs)
    swallowing = input("Do you have any kind of difficulity in swallowing? (Yes/No) : ")
    func(swallowing, user_inputs)
    chest_pain = input("Are you suffering from chest pain? (Yes/No) : ")
    func(chest_pain, user_inputs)
```
After obtaining a list of inputs from user, we take help of numpy array to reshape it and pass it to data. The main reason of doing this is because while training, we passed a 2D array.\
After this, we print output.

```python
    narry = np.asarray(user_inputs, dtype=int)
    reshaped_array = narry.reshape(1, -1)
    prediction = model.predict(reshaped_array)
    print("1 - High chances of lung cancer")
    print("0 - Low chances of lung cancer")
    print('\n\n\n\n\n')
    print("Prediction using Logistic regression : ", Logistic_Regression_model.predict(reshaped_array)[0])
    print("Prediction using Decision Tree : ", Decision_Tree_model.predict(reshaped_array)[0])
    print("Prediction using SVM Model : ", SVM_model.predict(reshaped_array)[0])
    print("Prediction using KNN Model : ", KNN_model.predict(reshaped_array)[0])
    print("Prediction using Random Forest Classifier : ", Random_Forest_Classifier.predict(reshaped_array)[0])
    print("Prediction using Naive Bayes Model : ", Naive_Bayes_model.predict(reshaped_array)[0])
if __name__ == "__main__":
    main()

```

Output samples: 
![Decision_Tree](https://user-images.githubusercontent.com/94124126/211165760-8105c99b-a183-4c3b-b199-f1dcadc5ef53.png)


