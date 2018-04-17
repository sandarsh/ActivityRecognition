from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
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
random_forest_classifier(feature,label)

