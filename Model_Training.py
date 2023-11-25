from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, make_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
import shap
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

# Define the parameter ranges for each classifier
param_grid = {
    'Random Forest': {'n_estimators': [1000], 
                      'max_depth': [None, 5, 10, 20, 30], 
                    #   'max_features': ['auto', 'sqrt', 'log2'], 
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]},
    'Linear SVM': {'C': [0.1, 1, 10, 100]},
    'Stochastic Gradient Descent': {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet']},
    'Logistic Regression': {'C': [0.1, 1, 10, 100]},
    'K-nearest neighbour': {'n_neighbors': [3, 5, 10, 15]},
    'Quadratic Discriminant Analysis': {},
    'AdaBoost': {'n_estimators': [10, 50, 100, 200]}
}

# Define the classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Linear SVM': SVC(kernel="linear",random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-nearest neighbour': KNeighborsClassifier(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'matthews_corrcoef': make_scorer(matthews_corrcoef),
    'f1': make_scorer(f1_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

with open('Var3PPred_v2_gridsearch_results.txt', 'w') as gs_results_file, open('Var3PPred_v2_crossval_results.txt', 'w') as cv_results_file:
    # For each classifier

    for name in classifiers:
        clf = classifiers[name]
        clf_param_grid = param_grid[name]

        # Perform GridSearchCV
        grid_search = GridSearchCV(clf, clf_param_grid, cv=3, n_jobs=10, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        print(name)
        # Make predictions with the best model
        y_pred = best_model.predict(X_test)
        y_probas = best_model.predict_proba(X_test)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_probas[:,1])
        roc_auc_test = auc(fpr_test, tpr_test)
        print(roc_auc_test)
        if name in ['Linear SVM', 'Stochastic Gradient Descent', 'Logistic Regression', ]:
            roc_auc_test = auc(fpr_test, tpr_test)
        
        plt.figure()
        # plt.plot(fpr_train, tpr_train, color='gray', lw=2, label='ROC curve (area = %0.2f) for CV' % roc_auc_train)
        plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f) for Test' % roc_auc_test)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(f'{name}_roc_curve.png', dpi=600)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot it using seaborn
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.figure()
        plt.savefig(f'{name}_confusion_matrix.png', dpi=600)


        if name in ['Random Forest', 'AdaBoost']:
            importances = best_model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in best_model], axis=0)
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            print("Feature ranking:")

            print("Feature ranking:")
            for f in range(X.shape[1]):
                print("%d. feature %s (%f)" % (f + 1, feature_name[indices[f]], importances[indices[f]]))

            # Plot the impurity-based feature importances of the forest
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), [feature_name[i] for i in indices], rotation='vertical')
            plt.xlim([-1, X.shape[1]])
            plt.show()
            plt.savefig(f'{name}_feature_importance.png', dpi=600)

        
        # Calculate metrics
        gs_results_file.write(f"\n{name}\n")
        gs_results_file.write("Best parameters: " + str(grid_search.best_params_) + "\n")
        gs_results_file.write("Accuracy: " + str(grid_search.best_score_) + "\n")
        gs_results_file.write(classification_report(y_test, y_pred) + "\n")
        gs_results_file.write("ROC AUC: " + str(roc_auc_score(y_test, y_pred)) + "\n")
        gs_results_file.write("ROC AUC: " + str(roc_auc_test) + "\n")
        gs_results_file.write("Balanced Accuracy: " + str(balanced_accuracy_score(y_test, y_pred)) + "\n")
        gs_results_file.write("Matthews Correlation Coefficient: " + str(matthews_corrcoef(y_test, y_pred)) + "\n")

        # Perform 20-fold cross-validation
        cv_scores = cross_val_score(best_model, X, y, cv=20, n_jobs=10)
        
        cv_scores = cross_validate(best_model, X, y, cv=20, scoring=scoring, n_jobs=10, return_train_score=True)

        # Write cross-validation results to file
        cv_results_file.write(f"\n{name}\n")
        for metric in scoring:
            cv_results_file.write(f"{metric} scores: " + ', '.join(f"{s:.2f}" for s in cv_scores[f'test_{metric}']) + "\n")
            cv_results_file.write(f"Mean {metric}: " + str(np.mean(cv_scores[f'test_{metric}'])) + "\n")
            cv_results_file.write(f"Standard deviation {metric}: " + str(np.std(cv_scores[f'test_{metric}'])) + "\n")

        # Save the best model
        joblib.dump(best_model, f'{name}_best_model2.pkl')
        



## SHAP Evaluation

shap.initjs()

# Compute SHAP values for training set
explainer = shap.TreeExplainer(clf.best_estimator_)
shap_values_train = explainer.shap_values(X_train)

# Make DataFrame for better visualization
X_train_df = pd.DataFrame(X_train, columns=feature_name)

# Compute SHAP interaction values for training set
shap_interaction_values_train = explainer.shap_interaction_values(X_train)

interaction_summary = np.sum(np.abs(shap_interaction_values_train), axis=0)
interaction_summary = np.sum(shap_interaction_values_train, axis=0)

# Plot SHAP values for training set
shap.summary_plot(shap_values_train, features=X_train_df, class_names=[ 'bening','patogenic'])#, plot_type='dot', show=False)
# plt.show()
shap.summary_plot(shap_interaction_values_train[1], features=X_train_df, sort=True, class_names=[ 'bening','patogenic'], class_inds=[0,1], max_display=100,  plot_type='bar', plot_size='auto')#, plot_type='dot', show=False)

expected_value = explainer.expected_value

shap.multioutput_decision_plot(expected_value, shap_values_train, 10, feature_names=feature_name)
# plt.show()
# shap.dependence_plot(feature_name[indices[0]], feature_names=feature_name, feature=shap_values_train, feature=X_train_df)

# shap.dependence_plot((feature_names[indices[0]], feature_names[indices[1]]), shap_values_train, X_train_df)
# shap_interaction_plot(shap_values_train, X_train)


# fig, axs = plt.subplots(22, 22, figsize=(22, 22))

# # Flatten the axs array to make iteration easier
# axs = axs.flatten()

# for i, feature_i in enumerate(feature_name):
#     for j, feature_j in enumerate(feature_name):
#         ax = axs[i*22 + j]
#         plt.sca(ax)  # Set the current Axes to ax
#         shap.dependence_plot(feature_i, shap_values_train[j],  X_train_df, interaction_index=None, show=False)

# plt.tight_layout()
# plt.show()


# Compute SHAP values for test set
shap_values_test = explainer.shap_values(X_test)




# Make DataFrame for better visualization
X_test_df = pd.DataFrame(X_test, columns=feature_name)

# Compute SHAP interaction values for test set
shap_interaction_values_test = explainer.shap_interaction_values(X_test)

# Plot SHAP values for test set
shap.summary_plot(shap_values_test, class_names=['bening','patogenic'], features=X_test_df)#, plot_type='dot', show=False)
# plt.show()
shap.summary_plot(shap_interaction_values_test[1], class_names=[ 'bening','patogenic'], sort=True, class_inds=[0,1], features=X_test_df, max_display=100,  plot_type='bar', plot_size='auto')#, plot_type='dot', show=False)
# plt.show()
shap.multioutput_decision_plot(expected_value, shap_values_test,  10, feature_names=feature_name)


# fig, axs = plt.subplots(22, 22, figsize=(22, 22))

# # Flatten the axs array to make iteration easier
# axs = axs.flatten()

# for i, feature_i in enumerate(feature_name):
#     for j, feature_j in enumerate(feature_name):
#         ax = axs[i*22 + j]
#         plt.sca(ax)  # Set the current Axes to ax
#         shap.dependence_plot(feature_i, shap_values_test[j],  X_test_df, interaction_index=None, show=False)

plt.tight_layout()
plt.show()



