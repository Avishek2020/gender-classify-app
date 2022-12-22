import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv("C:/Users/avish/Downloads/dataset.csv")
df.sex.replace({'F': 0, 'M': 1}, inplace= True)
Xfeatures = df['name']
print(df.head(12))
print(Xfeatures)

# CountVectorizer
count_vec = CountVectorizer()
X = count_vec.fit_transform(Xfeatures)
#print( count_vec.get_feature_names_out())
from sklearn.model_selection import train_test_split

X # Features
y = df.sex # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
print("Traing started...")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print("Traing finsihed...")


def predict(person_name: str) -> str:
    test_name = [person_name]
    vector = count_vec.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")


import joblib

# Save vectorizer For Gender Prediction
FinalVectorizer = open("gender_vectorizer.pkl", "wb")
joblib.dump(count_vec, FinalVectorizer)
FinalVectorizer.close()

# Save Model For Gender Prediction
naiveBayesModel = open("decisiontreemodel.pkl", "wb")
joblib.dump(clf, naiveBayesModel)
naiveBayesModel.close()
