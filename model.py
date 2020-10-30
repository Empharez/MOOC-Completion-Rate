import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate._ppoly import evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE


plt.style.use('ggplot')


data = pd.read_csv('new_data.csv')
df = pd.DataFrame(data)
df.dropna()

# shape
print('This data frame has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

# peek at data
df.sample(5)

# info
df.info()

# visualizations of age group, gender, disability, highest education
plt.figure(figsize=(10, 8))
plt.title('Distribution of Age group')
sns.distplot(df.age_group)

plt.figure(figsize=(10, 8))
plt.title('Distribution of Gender')
sns.distplot(df.gender)

plt.figure(figsize=(10, 8))
plt.title('Distribution of Disability')
sns.distplot(df.disability)

plt.figure(figsize=(10, 8))
plt.title('Distribution of Prev Educ')
sns.distplot(df.highest_education)


# dropout vs. completed course
counts = df.completed_mooc.value_counts()
completed = counts[1]
dropout = counts[0]
perc_completed = (completed / (completed + dropout)) * 100
perc_dropout = (dropout / (completed + dropout)) * 100
print(' {} completed their courses ({:.3f}%) and {} were dropouts ({:.3f}%).'.format(completed,
                                                                                    perc_completed,
                                                                                    dropout,
                                                                                    perc_dropout))

plt.figure(figsize=(8, 6))
sns.barplot(x=counts.index, y=counts)
plt.title('Count of incomplete vs. completed course')
plt.ylabel('Count')
plt.xlabel('Class (0:Incomplete, 1:Completed)')


corr = df.corr()
print(corr)

# heatmap
corr = df.corr()
plt.figure(figsize=(12, 10))
heat = sns.heatmap(data=corr)
plt.title('Heatmap of Correlation')


skew_ = df.skew()
print(skew_)

X = df[["gender", "highest_education", "age_group", "num_of_prev_attempts", "disability"]]
X_new = X.fillna(X.mean())

y = df["completed_mooc"]

"""counter = Counter(y)
print(counter)
oversample = SMOTE()
X_new, y = oversample.fit_resample(X_new, y)
counterOne = Counter(y)
print(counterOne)"""

"""training, evaluation = train_test_split(df, test_size=0.3, random_state=0)
validation, test = train_test_split(evaluation, test_size=0.5, random_state=0)



X_evaluation = evaluation[["gender", "highest_education", "age_group", "num_of_prev_attempts", "disability"]]
X_train = training[["gender", "highest_education", "age_group", "num_of_prev_attempts", "disability"]]
y_train = training['completed_mooc']

X_test = test[["gender", "highest_education", "age_group", "num_of_prev_attempts", "disability"]]
y_test = test['completed_mooc']

X_val = validation[["gender", "highest_education", "age_group", "num_of_prev_attempts", "disability"]]
y_val = validation['completed_mooc']
"""

X_train , X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3)
X_train = X_train.values
X_validation = X_test.values
y_train = y_train.values
y_validation = y_test.values

print('X_shapes:\n', 'X_train:', 'X_validation:\n', X_train.shape, X_validation.shape, '\n')
print('Y_shapes:\n', 'Y_train:', 'Y_validation:\n', y_train.shape, y_validation.shape)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, classification_report, roc_auc_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# visualizing RF
model = RandomForestClassifier(n_estimators=100, criterion='gini', max_features=3, min_samples_split=10, max_depth=90, min_samples_leaf=5)


"""from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
print(best_grid)"""
#grid_accuracy = evaluate(best_grid, test_features, test_labels)

# Train
model.fit(X_train, y_train)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# predict_train_with_if = model.predict(train_x_if)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
p_score = precision_score(y_test, y_pred, average='macro')
r_score = recall_score(y_test, y_pred, average='macro')
print(score, cm, p_score, r_score)


model_2 = SVC(decision_function_shape="ovr", kernel="rbf", C=10000)
model_2.fit(X_train , y_train)
svm_pred = model_2.predict(X_test)

score = accuracy_score(y_test, svm_pred)
cm = confusion_matrix(y_test, svm_pred)
p_score = precision_score(y_test, svm_pred, average='macro')
r_score = recall_score(y_test, svm_pred, average='macro')
print("svm score",score, cm, p_score, r_score)