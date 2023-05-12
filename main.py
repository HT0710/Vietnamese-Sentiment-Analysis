from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import pandas as pd
from utils import *
import os


# Download package
setup()


# Read dataset
dataset = pd.read_csv('IMDB.csv')

data, label = dataset['review'], dataset['sentiment']


# Multiprocessing - Đa luồng
limit = 0.8  # chạy 100% cpu dễ đứng máy
num_cpu = int(os.cpu_count() * limit)

with Pool(num_cpu) as pool:
    out = pool.map(remove_html, data)
    out = pool.map(remove_specials, out)
    out = pool.map(stemming, out)
    # out = pool.map(lemmatization, out)
    out = pool.map(remove_stopwords, out)
print('Preprocess - Done')


# Vectorization
X = ifidf(out, train=True, ngrams=(1, 1))
print('Words to Vector - Done')

# Label encoder
lb = LabelBinarizer()
y = lb.fit_transform(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model
LR = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, fit_intercept=False)
# DTR = DecisionTreeRegressor()

# Start training
print('Fit - Start')
model = LR.fit(X_train, y_train.ravel())
print('Fit - Done')


# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)

# Save result
# save_result(
#     prepocess=['stem'],
#     vectorizer='ifidf',
#     model='LogisticRegression',
#     accuracy=acc
# )
