# Lung Cancer Detection
Built this data-driven lung cancer detection system by employing multiple Machine Learning algorithms such as Logistic Regression, Decision Tree, SVM, KNN, Random Forest, and Naive Bayes on a dataset from [Kaggle](https://www.kaggle.com/), achieving a combined accuracy of 94.6183%, using python modules such as [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/),[scikit learn](https://scikit-learn.org/), [streamlit](https://streamlit.io/), [tensorflow](https://tensorflow.org/).
## Installation of required tools

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas, sklearn, streamlit, imblearn and tensorflow.
```python
pip install numpy
pip install pandas
pip install matplotlib
pip install sklearn
pip install streamlit
pip install imblearn
pip install tensorflow
```
## To run the file
```
    1. Download the source code, give it a name
    2. Open any terminal and locate the file
    3. Use the command "streamlit run file_name.py"
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
import tensorflow as tf
from tensorflow import keras
from imblearn.combine import SMOTETomek
```

## Classification techniques used
```
* Logistic Regression
* Decision Tree
* Support Vector Machine
* K - Nearest Neighbours
* Random Forest Classifier
* Naive Bayes
* Artificial Neural Networks
```
## Individual features 
```
Specific features in each model are described in the implementation.
```

## Prediction and accuracy
```
I made use of a function called accuracy_score, which is available in sklearn.metrics.
At first, I predicted data and stored it in variable ptrain, then compared it with original data (Y_train).
```
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

## Convert dictionary
```
This dictionary is used to map strings to integers and vice versa. Pleas have a look at the main function to better understand convert dictionary.
```
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

## Main function & Preparation of the data
```python
def main():
    #data retrieval
    data = pd.read_csv('D:/Programming/Streamlit/Lung_Cancer/cancer_data.csv')

    #plotting labels to check data
    # label_counts = data['LUNG_CANCER'].value_counts()
    # fig, ax = plt.subplots()
    # colors = ['Red', 'Green']
    # ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
    # st.pyplot(fig)


    data['LUNG_CANCER'] = data['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
    data['GENDER'] = data['GENDER'].replace({'M': 1, 'F': 0})

    X = data.drop(['LUNG_CANCER'], axis = 1)
    Y = data['LUNG_CANCER'].astype(int)


    #Handling imbalance
    smk = SMOTETomek(random_state = 42)
    X, Y = smk.fit_resample(X, Y)
    data = X
    data['LUNG_CANCER'] = Y
    X = data.drop(['LUNG_CANCER'], axis = 1).astype(int)
    Y = data['LUNG_CANCER'].astype(int)
    
    view_dataset = st.button("View dataset")
    if view_dataset:
        st.write(data)
```
```
The below part is used to handle input from user.
st.selectbox is a dropdown.
st.write, puts content on the screen.
```
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
# Input validation and splitting training and testing data
```python
    selected_model = st.selectbox("select your model : ", ("None", "Logistic Regression", "Decision Tree", "Support Vector Machine", "K Nearest Neighbour", "Random Forest", "Naive Bayes", "Artificial Neural Networks"))
    
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
## Artificial Neural Networks
```python
    elif selected_model == "Artificial Neural Networks":
            model = keras.Sequential([
                keras.layers.Dense(100, input_shape=(14,), activation = 'sigmoid'),
                keras.layers.Dense(100, activation = 'sigmoid'),
                keras.layers.Dense(100, activation = 'sigmoid'),
                keras.layers.Dense(2, activation = 'sigmoid'),
            ])
            model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
            with st.spinner('Please wait while model is running'):
                history = model.fit(X_train, Y_train, epochs = 120)
            predictArray = [reshaped_array]
            predictAns = model.predict(predictArray)
            st.write("Prediction using Artificial Neural networks : ", convert.get(np.argmax(predictAns)))
            view_train_accuracy = st.button("View train accuracy")
            if view_train_accuracy:
                accuracy = history.history['accuracy'][-1]
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
            view_test_accuracy = st.button("View test accuracy")
            if view_test_accuracy:
                loss, accuracy = model.evaluate(X_test, Y_test)
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



