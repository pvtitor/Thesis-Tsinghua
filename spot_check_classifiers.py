"""
This code is to spot check the best classifiers for the task of identifying the paraphrases.
"""

#Directory:
#C:\Users\pvtitor\Documents\1) Etudes\Tsinghua\M2\These\4) Python Code\Workspaces\Traitement Texte

"""
Steps:
- Visualize the data in different ways (using Pandas and pyplot)
- Put the data into a CSV file (IDs, features, label) => do this for train + test data
- Use KFold for test harness (folders = 10)
- Try all 6 classifiers on non-standardized data
- Standardize the data and try again the accuracy of our Algorithms
- Go through some algorithm tunning (hyper parameters)
"""

#Load libraries
import openpyxl
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


DEBUG = False
def fprint(string):
    if DEBUG:
        print(string)


# Load dataset
url_dataset = "C:\\Users\\pvtitor\\Documents\\1) Etudes\\Tsinghua\\M2\\These\\7) Results\\3) Features\\CSV Results.csv"
url_features = "C:\\Users\\pvtitor\\Documents\\1) Etudes\\Tsinghua\\M2\\These\\7) Results\\3) Features\\CSV Results_features.csv"
print("Reading dataset")
dataset = read_csv(url_dataset, header=0, sep = ";")
features = read_csv(url_features, header=0, sep = ";")
# print(dataset[0:10]), print()


"Analyse Data"

#shape
print(dataset.shape), print()

# types
set_option('display.max_rows', 4500)
print(dataset.dtypes), print()

# descriptions
set_option('precision', 3)
print(dataset.describe()), print()

# class distribution
# print(dataset.groupby(11).size()), print()


"Visualize Data"

# histograms
dataset.hist(layout=(3,3), sharex=True, sharey=True, bins = 20)
pyplot.show(), print()

# density
dataset.plot(kind="Density", subplots=True, layout=(3,3), sharex=True, sharey=True, legend=True, fontsize=1)
pyplot.show(), print()


# correlation matrix
names = ['LCSQ', 'LCST', 'BoW', 'SM', 'WER', 'PER', 'POSt', 'WuPal']
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(features.corr(), vmin=0, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


"Validation Dataset"

# Split-out validation dataset
array = dataset.values
X = array[:,0:8].astype(float)
print("X :"), print(X[0:10])
Y = array[:,8]
print("Y :"), print(numpy.transpose(Y[0:10]))
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


"Evaluate Algorithms"

# Test options and evaluation metric
num_folds = 10
seed = 7
# Defining the scorings
scorings = ['accuracy', 'precision', 'recall']
score_accuracy = ['accuracy']
score_precsion = ['precision']
score_recall = ['recall']

def get_results_scores(scores, models):
    results = []
    names = []
    for scoring in scorings:
        results.append(scoring)
        print(), print(scoring)
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results.mean())
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
    return results


# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Results of the models
print()
print("Results on non-standardized data")
results_no_standardized_data = get_results_scores(scorings, models)
print()



"Evaluate Algorithms : Standardize Data"

# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))

# Results of the models
print()
print("Results on standardized data")
results_standardized_data = get_results_scores(scorings, pipelines)
print()



# Compare Algorithms
scores = ['accuracy', 'precision', 'recall']

for scoring in scores:
    results_scaled = []
    names = []
    print(str(scoring) + "performances")
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results_scaled.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = pyplot.figure()
    fig.suptitle(str(scoring) + ' comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results_scaled)
    ax.set_xticklabels(names)
    pyplot.show(), print()
