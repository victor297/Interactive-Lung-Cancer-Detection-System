import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
    X, Y, test_size=0.3)


model = LogisticRegression(max_iter = 1000)
# model = LogisticRegression(max_iter = 100(by default))

model.fit(X_train, Y_train)

# ptrain = model.predict(X_train)
# accuracy = accuracy_score(Y_train, ptrain)
# print(accuracy)

# ptest = model.predict(X_test)
# accuracy = accuracy_score(Y_test, ptest)
# print(accuracy)


def func(str, l):
    if (str == 'Yes'):
        l.append(2)
    else:
        l.append(1)


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
    
    
    narry = np.asarray(user_inputs)
    reshaped_array = narry.reshape(1, -1)
    prediction = model.predict(reshaped_array)
    if(prediction == 1):
        print("High chances of lung cancer")
    else:
        print("Low chances of lung cancer")


if __name__ == "__main__":
    main()

