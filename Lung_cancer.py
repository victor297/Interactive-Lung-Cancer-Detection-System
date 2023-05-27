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

import streamlit as st
from time import sleep
st.title("Lung Cancer Detection")

convert = {
    'Yes': 2,
    'No': 1,
    'M': 1,
    'F': 0,
    1: 'High chances of Lung Cancer',
    0: 'Low chances of Lung Cancer'
}
# Yes - 2, No - 1


def main():
    data = pd.read_csv('D:\Programming\Streamlit\cancer_data.csv')
    data.loc[data['LUNG_CANCER'] == 'YES', 'label', ] = 1
    data.loc[data['LUNG_CANCER'] == 'NO', 'label', ] = 0

    data.loc[data['GENDER'] == 'M', 'GENDER', ] = 1
    data.loc[data['GENDER'] == 'F', 'GENDER', ] = 0

    X = data.drop(['LUNG_CANCER', 'label'], axis=1)
    Y = data['label'].astype(int)
    data = data.drop('LUNG_CANCER', axis=1)
    view_dataset = st.button("View dataset")
    if view_dataset:
        st.write(data)
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

    # model handling
    choosen_model = st.selectbox("Choose your model : ", ("None", "Logistic Regression", "Decision Tree", "Support Vector Machine",
                                                          "K Nearest Neigbours", "Random Forest", "Naive Bayes"))
    if not (gender and age and smoke and yellow_fingers and anxiety and chronic and fatigue and allergy and wheezing and alcohol and cough and breathing and swallowing and chest_pain) or None in user_inputs:
        pass
    elif not choosen_model or choosen_model == "None":
        st.write("Please choose one of the given models")
    else:
        # st.write('Model : ', choosen_model)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        # conversion of array
        narry = np.asarray(user_inputs, dtype=int)
        reshaped_array = narry.reshape(1, -1)

        if choosen_model == "Logistic Regression":
            solver = st.selectbox("Choose solver", ("None", "liblinear", "lbfgs", "sag", "saga"))
            if solver and solver != "None":
                if solver == "liblinear":
                    penalty = st.selectbox("Please choose penalty", ('l1', 'l2'))
                elif solver == "lbfgs" or solver == "sag":
                    penalty = st.selectbox("Please choose penalty", ('l2', 'none'))
                else:
                    penalty = st.selectbox("Please choose penalty", ('l1', 'l2', 'none'))
                Logistic_Regression_model = LogisticRegression(max_iter=1000, solver=solver, penalty=penalty)
                Logistic_Regression_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using Logistic regression : ", convert.get(
                    Logistic_Regression_model.predict(reshaped_array)[0]))

                view_train_accuracy = st.button("View train accuracy")
                if view_train_accuracy:
                    ptrain = Logistic_Regression_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")

                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = Logistic_Regression_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {choosen_model} {round(accuracy*100, 5)}%")

        elif choosen_model == "Decision Tree":
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
                    st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = Decision_Tree_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
                
        elif choosen_model == "Support Vector Machine":
            # svm_model = SVC() #By default Radial - Basis function kernel
            choosen_kernel = st.selectbox(
                "Please select a kernel", ("None", "linear", "rbf", "poly"))
            if choosen_kernel and choosen_kernel != "None":
                svm_model = SVC(kernel=choosen_kernel)
                svm_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using SVM Model : ", convert.get(
                    svm_model.predict(reshaped_array)[0]))
                view_train_accuracy = st.button("View train accuracy")
                if view_train_accuracy:
                    ptrain = svm_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
                view_test_accuracy = st.button("View test accuracy")                
                if view_test_accuracy:
                    ptest = svm_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {choosen_model} is {round(accuracy*100, 5)}%")

        elif choosen_model == "K Nearest Neigbours":
            # knn_model = KNeighborsClassifier(value of k) #By default, value of k is 5
            weights = st.selectbox("Choose weights", ("Select", "uniform", "distance"))
            if weights != "Select":
                kval = st.slider("Please choose the value of k", 0, len(X_train.values))
                if kval:
                    knn_model = KNeighborsClassifier(n_neighbors = kval, weights=weights)
                    knn_model.fit(X_train.values, Y_train.values)
                    st.write("Prediction using KNN Model : ", convert.get(
                        knn_model.predict(reshaped_array)[0]), " with k as ", kval)
                    view_train_accuracy = st.button("View train accuracy")
                    if view_train_accuracy:
                        ptrain = knn_model.predict(X_train.values)
                        accuracy = accuracy_score(Y_train, ptrain)
                        st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
                    view_test_accuracy = st.button("View test accuracy")                
                    if view_test_accuracy:
                        ptest = knn_model.predict(X_test.values)
                        accuracy = accuracy_score(Y_test, ptest)
                        st.write(f"Test accuracy for {choosen_model} is {round(accuracy*100, 5)}%")

        elif choosen_model == "Random Forest":
            n = st.slider("Choose number of estimators", 1, 5000)
            criterion = st.selectbox("Criterion", ('gini', 'entropy'))
            Random_Forest_Classifier = RandomForestClassifier(n_estimators = n, criterion=criterion)
            Random_Forest_Classifier.fit(X_train.values, Y_train.values)
            st.write("Prediction using Random Forest Classifier : ", convert.get(
                Random_Forest_Classifier.predict(reshaped_array)[0]))
            view_train_accuracy = st.button("View train accuracy")
            if view_train_accuracy:
                ptrain = Random_Forest_Classifier.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
            view_test_accuracy = st.button("View test accuracy")                
            if view_test_accuracy:
                ptest = Random_Forest_Classifier.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
            
        elif choosen_model == "Naive Bayes":
            Naive_Bayes_model = GaussianNB()
            Naive_Bayes_model.fit(X_train.values, Y_train.values)
            st.write("Prediction using Naive Bayes Model : ", convert.get(
                Naive_Bayes_model.predict(reshaped_array)[0]))
            view_train_accuracy = st.button("View train accuracy")
            if view_train_accuracy:
                ptrain = Naive_Bayes_model.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {choosen_model} is {round(accuracy*100, 5)}%")
            view_test_accuracy = st.button("View test accuracy")                
            if view_test_accuracy:
                ptest = Naive_Bayes_model.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {choosen_model} is {round(accuracy*100, 5)}%")


if __name__ == "__main__":
    main()
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
