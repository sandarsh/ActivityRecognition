from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

input_folder = 'Features_1p5sec/'




def random_forest_classifier(features,classification):

    features_train, features_test, classification_train, classification_test = train_test_split(features,classification,test_size=0.20,random_state=0)
    classifier = RandomForestClassifier(n_estimators=600)

    classifier = classifier.fit(features_train,classification_train)
    class_prediction = classifier.predict(features_test)

    x = list(range(len(class_prediction)))


    print(class_prediction)

    plt.scatter(x,class_prediction,color='red')

    plt.plot(x,classification_test,color='blue')


    plt.xticks(())
    plt.yticks(())

    # plt.show()



    print("error mse")
    print(mean_squared_error(classification_test,class_prediction))
    print("variance score: 1 is perfect prediction")
    print(r2_score(classification_test,class_prediction))

    print(classification_report(classification_test,class_prediction))



# df = pd.read_csv('Features/6_ws_3_opc_10.features.csv',sep=',',skiprows=[0])
#
# print df.shape
# label = df[df.columns[-1]]
# feature = df.drop([df.columns[-1]],axis=1).values
# print feature.shape
# print label.value_counts()


def supervised_learning(features,classification,algo):
    features_train, features_test, classification_train, classification_test = train_test_split(features,classification,test_size=0.20,random_state=0)
    classifier = RandomForestClassifier(n_estimators=600)

    if algo == 'NB':
        classifier = GaussianNB()
    elif algo == 'KNN':
        print("KNN")
        classifier = KNeighborsClassifier()
    elif algo == 'SVM':
        print("SVM")
        classifier = svm.SVC(kernel='linear')
    elif algo == 'SVMPOLY':
        print("SVMPOLY")
        classifier = svm.SVC(kernel='poly', degree=3)
    elif algo == 'SVMRBF':
        print("SVMRBF")
        classifier = svm.SVC(kernel='rbf')
    elif algo == 'DT':
        print("DT")
        classifier = DecisionTreeClassifier()
    elif algo=='LOGISTIC':
        print("LOGISTIC")
        classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    elif algo=='RF':
        print("RF")
        classifier = RandomForestClassifier(n_estimators=600)

    classifier = classifier.fit(features_train, classification_train)
    class_prediction = classifier.predict(features_test)
    x = list(range(len(class_prediction)))
    print(class_prediction)
    plt.scatter(x,class_prediction,color='red')
    plt.plot(x,classification_test,color='blue')
    plt.xticks(())
    plt.yticks(())

    # plt.show()
    print("error mse")
    print(mean_squared_error(classification_test,class_prediction))
    print("variance score: 1 is perfect prediction")
    print(r2_score(classification_test,class_prediction))
    print(classification_report(classification_test,class_prediction))



df = pd.DataFrame()
for dir, file, files in os.walk(input_folder):
    for each in files:
        if '.features.csv' in each:
            print(each)
            df = df.append(pd.read_csv(input_folder + each, sep=','), ignore_index=True)
print(df.shape)
label = df['label']
feature = df.drop(['label'], axis=1)




print(feature.shape)
print(label.value_counts())

p=0.8
# corr = VarianceThreshold(p)
# # feature = corr.fit_transform(feature)

def remove_corr_features(feature):
    corr_matrix = feature.corr().abs()

    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # high_corr = [c for c in corr_values.columns if any(corr_values[c] > 0.8)]
    high_corr = []
    count = 0
    print("corelation shape is")
    print(corr_values)

    for c in corr_values.columns:
        if any(corr_values[c] > 0.8):
            high_corr.append(count)
        count += 1
    print(len(feature.columns[high_corr]), (feature.columns[high_corr]))

    feature = feature.drop(feature.columns[high_corr], axis=1)
    return feature

feature = remove_corr_features(feature)

print(feature.shape)

# random_forest_classifier(feature,label)
# supervised_learning(feature,label,'KNN')

