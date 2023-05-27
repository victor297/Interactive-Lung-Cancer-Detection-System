# Lung Cancer Detection
Built this data-driven lung cancer detection system by employing multiple Machine Learning algorithms such as Logistic Regression, Decision Tree, SVM, KNN, Random Forest, and Naive Bayes on a dataset from [Kaggle](https://www.kaggle.com/), achieving a combined accuracy of 94.6183%, using python modules such as [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit learn](https://scikit-learn.org/), [streamlit](https://streamlit.io/).
## Installation of required tools

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas, sklearn.
```python
pip install numpy
pip install pandas
pip install sklearn
pip install streamlit
```
## To run the file
```
    1. Download the source code, give it a name such as "LCD.py"
    2. Open the terminal and locate the file
    3. Use the command "streamlit run LCD.py"
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
import streamlit
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
## Individual features 
Specific features in each model are described in the implementation

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

# Convert dictionary
This dictionary is used to convert string to corresponding integer.\
```python
convert = {
    'Yes': 2,
    'No': 1,
    'M': 1,
    'F': 0,
    1: 'High chances of Lung Cancer',
    0: 'Low chances of Lung Cancer'
}
```
## The main function
In the main function, we take input for every value from user, then convert that respond to an integer using dictionary.

## Preparation of data
```python
def main():
    data = pd.read_csv('D:/MyProjects/Lung_Cancer_Detection/cancer_data.csv')
    data.loc[data['LUNG_CANCER'] == 'YES', 'label', ] = 1
    data.loc[data['LUNG_CANCER'] == 'NO', 'label', ] = 0

    data.loc[data['GENDER'] == 'M', 'GENDER', ] = 1
    data.loc[data['GENDER'] == 'F', 'GENDER', ] = 0

    X = data.drop(['LUNG_CANCER', 'label'], axis=1)
    Y = data['label'].astype(int)
    data = data.drop('LUNG_CANCER', axis=1)
    view_dataset = st.button("View dataset") #button from streamlit
    if view_dataset: # to check whether the button is clicked
        st.write(data)
```
The below part is used to handle input from user. \
st.selectbox is a dropdown.
st.write, puts content on the screen.
```python
    user_inputs = []

    gender = st.selectbox("Please select your gender : ", ("Select", "M", "F"))
    user_inputs.append(convert.get(gender))
    age = st.number_input("Please enter your age : ", value=0, step=1)
    user_inputs.append(age)
    smoke = st.selectbox("Do you smoke?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(smoke))
    yellow_fingers = st.selectbox(
        "Do you have yellow fingers?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(yellow_fingers))
    anxiety = st.selectbox("Do you have anxiety issues?",
                           ("Select", "Yes", "No"))
    user_inputs.append(convert.get(anxiety))
    chronic = st.selectbox(
        "Do you have any chronic diseases?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(chronic))
    fatigue = st.selectbox("Do you experience fatigue?",
                           ("Select", "Yes", "No"))
    user_inputs.append(convert.get(fatigue))
    allergy = st.selectbox("Do you have any allergies?",
                           ("Select", "Yes", "No"))
    user_inputs.append(convert.get(allergy))
    wheezing = st.selectbox("Do you wheeze?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(wheezing))
    alcohol = st.selectbox("Do you consume alcohol?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(alcohol))
    cough = st.selectbox("Do you cough on a regular basis?",
                         ("Select", "Yes", "No"))
    user_inputs.append(convert.get(cough))
    breathing = st.selectbox(
        "Do you have any kind of difficulity in breathing?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(breathing))
    swallowing = st.selectbox(
        "Do you have any kind of difficulity in swallowing?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(swallowing))
    chest_pain = st.selectbox(
        "Are you suffering from chest pain?", ("Select", "Yes", "No"))
    user_inputs.append(convert.get(chest_pain))
```
# Input validation and training
```python
    selected_model = st.selectbox("select your model : ", ("None", "Logistic Regression", "Decision Tree", "Support Vector Machine", "K Nearest Neighbour", "Random Forest", "Naive Bayes"))
    
    # handle errors in input
    if not (gender and age and smoke and yellow_fingers and anxiety and chronic and fatigue and allergy and wheezing and alcohol and cough and breathing and swallowing and chest_pain) or None in user_inputs:
        pass 
    elif not selected_model or selected_model == "None": 
        st.write("Please select one of the given models")
    else:
        # st.write('Model : ', selected_model)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        # conversion of array
        narry = np.asarray(user_inputs, dtype=int)
        reshaped_array = narry.reshape(1, -1)
```
## Logistic Regression
```python
        if selected_model == "Logistic Regression":
            solver = st.selectbox("select solver", ("None", "liblinear", "lbfgs", "sag", "saga"))
            if solver and solver != "None":
                if solver == "liblinear":
                    penalty = st.selectbox("Please select penalty", ('l1', 'l2'))
                elif solver == "lbfgs" or solver == "sag":
                    penalty = st.selectbox("Please select penalty", ('l2', 'none'))
                else:
                    penalty = st.selectbox("Please select penalty", ('l1', 'l2', 'none'))
                Logistic_Regression_model = LogisticRegression(max_iter=1000, solver=solver, penalty=penalty)
                Logistic_Regression_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using Logistic regression : ", convert.get(
                    Logistic_Regression_model.predict(reshaped_array)[0]))

                view_train_accuracy = st.button("View train accuracy")
                if view_train_accuracy:
                    ptrain = Logistic_Regression_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")

                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = Logistic_Regression_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} {round(accuracy*100, 5)}%")
```
## Decision Tree
```python
        elif selected_model == "Decision Tree":
            criterion = st.selectbox("Criterion", ('Select', 'gini', 'entropy'))
            if criterion != 'Select':
                Decision_Tree_model = DecisionTreeClassifier(criterion=criterion)
                Decision_Tree_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using Decision Tree : ", convert.get(
                    Decision_Tree_model.predict(reshaped_array)[0]))
                view_tree = st.button("View Tree")
                if view_tree:
                    # plt.figure(figsize=(width, height))
                    with st.spinner('Please wait while the decision tree is being loaded...'):
                        plt.figure(figsize=(50, 10))
                        plot_tree(decision_tree=Decision_Tree_model, filled=True,
                            feature_names=X.columns, rounded=True)
                        st.pyplot(plt)
                view_train_accuracy = st.button("View train accuracy")
                if view_train_accuracy:
                    ptrain = Decision_Tree_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = Decision_Tree_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
```
## Support Vector Machine
```python
        elif selected_model == "Support Vector Machine":
            # svm_model = SVC() #By default Radial - Basis function kernel
            selected_kernel = st.selectbox(
                "Please select a kernel", ("None", "linear", "rbf", "poly"))
            if selected_kernel and selected_kernel != "None":
                svm_model = SVC(kernel=selected_kernel)
                svm_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using SVM Model : ", convert.get(
                    svm_model.predict(reshaped_array)[0]))
                view_train_accuracy = st.button("View train accuracy")
                if view_train_accuracy:
                    ptrain = svm_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = svm_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
```
## K Nearest Neighbours
```python
        elif selected_model == "K Nearest Neighbour":
            # knn_model = KNeighborsClassifier(value of k) #By default, value of k is 5
            weights = st.selectbox("select weights", ("Select", "uniform", "distance"))
            if weights != "Select":
                kval = st.slider("Please select the value of k", 0, len(X_train.values))
                if kval:
                    knn_model = KNeighborsClassifier(n_neighbors = kval, weights=weights)
                    knn_model.fit(X_train.values, Y_train.values)
                    st.write("Prediction using KNN Model : ", convert.get(
                        knn_model.predict(reshaped_array)[0]), " with k as ", kval)
                    view_train_accuracy = st.button("View train accuracy")
                    if view_train_accuracy:
                        ptrain = knn_model.predict(X_train.values)
                        accuracy = accuracy_score(Y_train, ptrain)
                        st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                    view_test_accuracy = st.button("View test accuracy")                
                    if view_test_accuracy:
                        ptest = knn_model.predict(X_test.values)
                        accuracy = accuracy_score(Y_test, ptest)
                        st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
```
## Random Forest
```python
        elif selected_model == "Random Forest":
            n = st.slider("select number of estimators", 1, 5000)
            criterion = st.selectbox("Criterion", ('gini', 'entropy'))
            Random_Forest_Classifier = RandomForestClassifier(n_estimators = n, criterion=criterion)
            Random_Forest_Classifier.fit(X_train.values, Y_train.values)
            st.write("Prediction using Random Forest Classifier : ", convert.get(
                Random_Forest_Classifier.predict(reshaped_array)[0]))
            view_train_accuracy = st.button("View train accuracy")
            if view_train_accuracy:
                ptrain = Random_Forest_Classifier.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
            view_test_accuracy = st.button("View test accuracy")                
            if view_test_accuracy:
                ptest = Random_Forest_Classifier.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
```
## Naive Bayes
```python
        elif selected_model == "Naive Bayes":
            Naive_Bayes_model = GaussianNB()
            Naive_Bayes_model.fit(X_train.values, Y_train.values)
            st.write("Prediction using Naive Bayes Model : ", convert.get(
                Naive_Bayes_model.predict(reshaped_array)[0]))
            view_train_accuracy = st.button("View train accuracy")
            if view_train_accuracy:
                ptrain = Naive_Bayes_model.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
            view_test_accuracy = st.button("View test accuracy")                
            if view_test_accuracy:
                ptest = Naive_Bayes_model.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
```
## Calling the main function
```python
if __name__ == "__main__":
    main()
```
## Establishing a footer
```python
footer = """
    <style>
    a:hover
    {
        opacity:50%;
    }
    .footer 
    {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        text-align: center;
    }
    </style>
    <div class="footer">
        <p> Developed with ❤ by <a style='display: block; text-align: center;' href="https://iamssuraj.netlify.app/" target="_blank"> Suraj Sanganbhatla </a> </p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
```
Output Samples: 
![Screenshot (997)](https://github.com/iamssuraj/Data-Driven-Early-Diagnosis-of-Lung-Cancer-Using-Advanced-Machine-Learning-Models/assets/94124126/f00993bf-1f10-4c19-bcda-744cb2beedf0)

![Screenshot (998)](https://github.com/iamssuraj/Data-Driven-Early-Diagnosis-of-Lung-Cancer-Using-Advanced-Machine-Learning-Models/assets/94124126/e4b36be7-cbf0-4f43-b3bb-9112712633da)

![Screenshot (999)](https://github.com/iamssuraj/Data-Driven-Early-Diagnosis-of-Lung-Cancer-Using-Advanced-Machine-Learning-Models/assets/94124126/b4e98413-f99e-4549-9635-fda7bf5e4201)



