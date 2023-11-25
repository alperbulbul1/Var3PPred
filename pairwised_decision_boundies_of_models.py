from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from itertools import combinations
import matplotlib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
matplotlib.rcParams.update({'font.size': 22})
from fancyimpute import IterativeImputer

df = pd.read_csv('train_data.csv')

# Initialize the MICE imputer
mice_imputer = IterativeImputer(random_state=123)

df_new = df.drop('Name', axis=1)
# Apply the imputer
df = df_new.copy(deep=True)
df.iloc[:, :] = mice_imputer.fit_transform(df_new)


import seaborn as sns
sns.set(font_size=1.4)

sns.set_theme(style="ticks")
df.loc[df['Label'] == 0, 'Label'] = 'Bening'
df.loc[df['Label'] == 1, 'Label'] = 'Pathogenic'
sns.pairplot(df, hue='Label', kind='reg')
plt.savefig('Feature_comparision.png')

smote = SMOTE(random_state=123)

y = df.pop('Label')

feature_name = df.columns.tolist()
X = df.values

# Fit and apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)
y = y_resampled

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X = X_resampled
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

param_grid = {
    'Random Forest': {'n_estimators': [1000],
                      'max_depth': [None, 5, 10, 20, 30],
                    #   'max_features': ['auto', 'sqrt', 'log2'],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]},
    'Linear SVM': {'C': [0.1, 1, 10, 100]},
    # 'Stochastic Gradient Descent': {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet']},
    'Logistic Regression': {'C': [0.1, 1, 10, 100]},
    'K-nearest neighbour': {'n_neighbors': [3, 5, 10, 15]},
    'Quadratic Discriminant Analysis': {},
    'AdaBoost': {'n_estimators': [10, 50, 100, 200]}
}

# Define the classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Linear SVM': SVC(kernel="linear",random_state=42, probability=True),
    # 'Stochastic Gradient Descent': SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-nearest neighbour': KNeighborsClassifier(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

h = .02
comb = combinations(feature_name, 2)
comb_list = list(comb)
print(comb_list)
df_X_resampled = pd.DataFrame(X_resampled, columns=feature_name)
for idx, (feature1, feature2) in enumerate(comb_list[0:2]):
    X_resampled_split = df_X_resampled[[feature1, feature2]]
    X = X_resampled_split.values
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    # ax_al = plt.subplot
    print(X)
    datasets = [linearly_separable]

    figure = plt.figure(figsize=(27, 45))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        # X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.2)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        print(np.c_[xx.ravel(), yy.ravel()])

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(comb_list), len(list(classifiers.values())) + 1, i + len(classifiers) * idx)
        # ax = plt.subplot(len(datasets), len(list(classifiers.values())) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(list(classifiers.keys()), list(classifiers.values())):
            ax = plt.subplot(len(datasets), len(list(classifiers.values())) + 1, i)
            clf_param_grid = param_grid[name]
            grid_search = GridSearchCV(clf, clf_param_grid, cv=3, n_jobs=10, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            # clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=22, horizontalalignment='right', color='#8e7cc3')
            i += 1

    plt.tight_layout()
    plt.show()
# plt.savefig('Feature_model_comparison.png', dpi=600)