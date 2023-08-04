import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Preparing the data
df = pd.read_csv('./data/mail_data.csv', header=None)

# Assign the first row values to column names
df.columns = df.iloc[0]
df = df.iloc[1:]  # Remove the first row, as it's already assigned as column names

df.reset_index(inplace=True, drop=True)
# df = df.where(pd.notnull(df), '')

# replace ham with 0 and spa with 1
df.loc[df['Category'] == 'ham', 'Category'] = 1
df.loc[df['Category'] == 'spam', 'Category'] = 0


# Split data to train and test sets

X = df['Message']
Y = df['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# Perform TF-IDF:
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# Create a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

Y_predicted_training = model.predict(X_train_features)
Y_predicted_test = model.predict(X_test_features)
accuracy_training = accuracy_score(Y_train, Y_predicted_training)
accuracy_test = accuracy_score(Y_test, Y_predicted_test)
cm = confusion_matrix(Y_test, Y_predicted_test)

print("Accuracy on the training data:", accuracy_training)
print("Accuracy on the testing data:", accuracy_test)
print('Confusion matrix:\n\n', cm)
print()


#Turns out, that if we change threshold from 0.5 to 0.8, our metrics will return much better results
Y_pred_train__threshold_085 = (model.predict_proba(X_train_features)[:, 1] >= 0.85).astype(int)
Y_pred_threshold_085 = (model.predict_proba(X_test_features)[:, 1] >= 0.85).astype(int)
cm2 = confusion_matrix(Y_test, Y_pred_threshold_085)

print("Accuracy on the training data for threshold = 0.85:", accuracy_score(Y_train, Y_pred_train__threshold_085))
print("Accuracy on the testing data for threshold = 0.85:", accuracy_score(Y_test, Y_pred_threshold_085))
print('Confusion matrix for for threshold = 0.85:\n\n', cm2)
