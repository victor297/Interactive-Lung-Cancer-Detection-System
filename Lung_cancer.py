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

data = pd.read_csv('D:/MyProjects/Lung_Cancer_Detection/cancer_data.csv')

'''
In inbuilt data,
YES --> 2
NO --> 1
'''

data.loc[data['LUNG_CANCER'] == 'YES', 'label', ] = 1
data.loc[data['LUNG_CANCER'] == 'NO', 'label', ] = 0


data.loc[data['GENDER'] == 'M', 'GENDER', ] = 1
data.loc[data['GENDER'] == 'F', 'GENDER', ] = 0

X = data.drop(['LUNG_CANCER', 'label'], axis=1)
Y = data['label'].astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state = 42)


#---------------------------------------------------------------------------------

# Logisitic Regression Model
Logistic_Regression_model = LogisticRegression(max_iter=1000)
Logistic_Regression_model.fit(X_train.values, Y_train.values)

# ptrain = Logistic_Regression_model.predict(X_train.values)
# accuracy = accuracy_score(Y_train, ptrain)
# print(accuracy)

print('Test Accuracy for Logisitic Regression Model : ', end = '')
ptest = Logistic_Regression_model.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)

#---------------------------------------------------------------------------------

# Decision Tree
Decision_Tree_model = DecisionTreeClassifier()
Decision_Tree_model.fit(X_train.values, Y_train.values)

print('Test Accuracy for Decision Tree Model : ', end = '')
ptest = Decision_Tree_model.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)

# Plotting the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(decision_tree = Decision_Tree_model, filled=True, feature_names=X.columns, rounded = True)
plt.show()

#---------------------------------------------------------------------------------

# Support Vector Machine Model
# svm_model = SVC() #By default Radial - basis function kernel
svm_model = SVC(kernel = 'linear')
svm_model.fit(X_train.values, Y_train.values)

print('Test Accuracy for SVM Model : ', end = '')
ptest = svm_model.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)

#---------------------------------------------------------------------------------

# K - Nearest Neighbours Model

# knn_model = KNeighborsClassifier(value of k) #By default, value of k is 5
knn_model = KNeighborsClassifier(5) 
knn_model.fit(X_train.values, Y_train.values)

print('Test Accuracy for KNN Model : ', end = '')
ptest = knn_model.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)

#---------------------------------------------------------------------------------

# Random Forest Classifier
Random_Forest_Classifier = RandomForestClassifier()
Random_Forest_Classifier.fit(X_train.values, Y_train.values)

print('Test Accuracy for Random Forest Classifier Model : ', end = '')
ptest = Random_Forest_Classifier.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)

#---------------------------------------------------------------------------------


# Naive Bayes Model
Naive_Bayes_model = GaussianNB()
Naive_Bayes_model.fit(X_train.values, Y_train.values)

print('Test Accuracy for Naive Bayes Model : ', end = '')
ptest = Naive_Bayes_model.predict(X_test.values)
accuracy = accuracy_score(Y_test, ptest)
print(accuracy)


user_inputs = []
def func(str, user_inputs):
    if (str == 'Yes'):
        user_inputs.append(2)
    else:
        user_inputs.append(1)

def main():
    print('\n')
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
    breathing = input(
        "Do you have any kind of difficulity in breathing? (Yes/No) : ")
    func(breathing, user_inputs)
    swallowing = input(
        "Do you have any kind of difficulity in swallowing? (Yes/No) : ")
    func(swallowing, user_inputs)
    chest_pain = input("Are you suffering from chest pain? (Yes/No) : ")
    func(chest_pain, user_inputs)

    narry = np.asarray(user_inputs, dtype=int)
    reshaped_array = narry.reshape(1, -1)
    print('\n')
    print('\n')
    print("1 - High chances of lung cancer")
    print("0 - Low chances of lung cancer")
    print('\n')
    print("Prediction using Logistic regression : ", Logistic_Regression_model.predict(reshaped_array)[0])
    print("Prediction using Decision Tree : ", Decision_Tree_model.predict(reshaped_array)[0])
    print("Prediction using SVM Model : ", svm_model.predict(reshaped_array)[0])
    print("Prediction using KNN Model : ", knn_model.predict(reshaped_array)[0])
    print("Prediction using Random Forest Classifier : ", Random_Forest_Classifier.predict(reshaped_array)[0])
    print("Prediction using Naive Bayes Model : ", Naive_Bayes_model.predict(reshaped_array)[0])

if __name__ == "__main__":
    main()

