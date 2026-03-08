## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
import joblib

## skelarn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## skelarn -- models
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

## sklearn -- metrics
from sklearn.metrics import f1_score, confusion_matrix

os.makedirs('models', exist_ok=True)

legacy_plots = ['without-imbalance.png', 'with-class-weights.png', 'with-SMOTE.png']
for plot_file in legacy_plots:
    if os.path.exists(plot_file):
        os.remove(plot_file)

## --------------------- Data Preparation ---------------------------- ##

## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'dataset.csv')
df = pd.read_csv(TRAIN_PATH)

## Drop first 3 features
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

## Filtering using Age Feature using threshold
df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)


## To features and target
X = df.drop(['Exited'], axis=1)
y = df['Exited']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)


## --------------------- Data Processing ---------------------------- ##

## Slice the lists
num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))


## For Numerical
num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])



## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)
                                ])

## apply
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


## --------------------- Impalancing ---------------------------- ##

## 1. use algorithm without taking the effect of imbalancing

## 2. prepare class_weights for solving imbalance dataset
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  ## normalizing


dict_weights = {}
for i in range(2):  ## 2 classes (0, 1)
    dict_weights[i] = vals_count[i]

## 3. Using SMOTE for over sampling
over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)


## --------------------- Modeling ---------------------------- ##

## Clear metrics.txt file at the beginning
with open('metrics.txt', 'w') as f:
    pass


def train_model(X_train, y_train, plot_name='', class_weight=None):
    """ A function to train model given the required train data """
    
    global clf_name

    clf = LogisticRegression(C=2, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test_final)
    
    ## Using f1_score
    score_train = f1_score(y_train, y_pred_train)
    score_test = f1_score(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test)
    
    clf_name = clf.__class__.__name__

    ## Write scores to a file
    with open('metrics.txt', 'a') as f:
            f.write(f'{clf_name} {plot_name}\n')
            f.write(f"F1-score of Training is: {score_train*100:.2f} %\n")
            f.write(f"F1-Score of Validation is: {score_test*100:.2f} %\n")
            f.write('----'*10 + '\n')


    joblib.dump(clf, os.path.join(os.getcwd(), 'models', f'{clf_name}-{plot_name}.pkl'))

    return cm, clf_name


## 1. without considering the imabalancing data
conf_mats = []
cm, clf_name = train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance', class_weight=None)
conf_mats.append(('without-imbalance', cm))

## 2. with considering the imabalancing data using class_weights
cm, clf_name = train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', class_weight=dict_weights)
conf_mats.append(('with-class-weights', cm))

## 3. with considering the imabalancing data using oversampled data (SMOTE)
cm, clf_name = train_model(X_train=X_train_resmapled, y_train=y_train_resampled, plot_name=f'with-SMOTE', class_weight=None)
conf_mats.append(('with-SMOTE', cm))


## Combine all conf matrix in one figure
fig, axes = plt.subplots(1, len(conf_mats), figsize=(15, 5))

for ax, (title, cm) in zip(axes, conf_mats):
    sns.heatmap(cm, annot=True, cbar=False, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    ax.set_yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=300)
plt.close(fig)
