import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

data=pd.read_csv("twitter_training.csv")

data.columns=["ID","Topic","sentiment","tweet"]

data=data.dropna(subset=["tweet"])

print(data.head())
print(data.info())

sns.countplot(x='sentiment',data=data)
plt.title("distribution  map")
plt.show()

X=data["tweet"]
y=data["sentiment"]

vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=LogisticRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)

f1=f1_score(y_test,predictions,average="weighted")
print("f1 score:",f1)

score=accuracy_score(y_test,predictions)
print("accuracy score:",score)