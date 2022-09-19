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
## Implementation of Model
Since we are dealing with a classification problem, we use Logistic Regression. By default in lbfgs, the maximum itertaions are 100. If there is a need for increase in iterations, we can pass a parameter 'max_iter' as shown below.
```python
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
```

## Prediction and accuracy
From sklearn.metrics, we have already imported accuracy_score, which will be used here to calculate how accurately our model is able to predict data.\
At first we calculate the predicted data and store it in variable ptrain, then compare it with original data (Y_train).
```python
ptrain = model.predict(X_train)
accuracy = accuracy_score(Y_train, ptrain)
print(accuracy)
```

The above task is performed again on test data, to avoid overfitting.
```python
ptest = model.predict(X_test)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)
```

## Convert function 
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
    if (prediction == 1):
        print("High chances of lung cancer")
    else:
        print("Low chances of lung cancer")
if __name__ == "__main__":
    main()

```

Output samples: 
![Output_Screenshot](https://user-images.githubusercontent.com/94124126/191048414-05d75c9b-9278-4714-aa42-c5f621b87cee.png)
![Screenshot2](https://user-images.githubusercontent.com/94124126/191049025-612d440f-8b81-43b9-bd56-45202821b8e2.png)


