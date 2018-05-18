# Thesis-Tsinghua

### Development of a Machine Learning Algorithm to compare RAMS Standards Documents ###
=======================================================

This project contains the main codes implemented (with python) for my master thesis at Tsinghua University (Beijing).

This research aims to compare RAMS Standards Documents by:
- using natural languale processing tools
- implementing lexical, semantic and syntactic features 
- developping a machine learning classifier to train our model

To train and test my algorithm, I made use of the well-known Microsoft Reseach Paraphrase Corpus (MSRPC).
(available at https://www.microsoft.com/en-us/download/details.aspx?id=52398)

My model provides already promising results, similar to the state of the art techniques for the Paraphrase Identification task, implemented with other features and classifiers which also make use of the MSRPC dataset.
(confer: https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art))

Performances achieved:
- Accuracy  = 74.4%
- F-score   = 82.2%
=======================================================

### Feature Extraction ###

The "feature extraction" is the main code of this research.
It implements different lexical (6), syntactic (1) and semantic (1) features to caracterize numerically the similarity of any pair of sentences. This is the core of the algorithm, because the vector of features is the main input for any decision-making algorithm that I may use afterwards to decide whether the sentences are paraphrase of each other or not.
It can be improved in several ways:
- improving the features themselves (for example making the the lexical features more performant by already taking into account some syntactic elements)
- changing the way of calculation of the similarity between a pair of sentences for the already implemented features
- adding some new features that will bring along new understanding of the pairs of sentences (for instance *word alignment*, *cardinal attribute*, or other semantic feature that will compare the meaning of groups of words directly).
=======================================================

### Spot-Check Classifiers ###

This code aims to quickly review the main machine learning classifiers (linear and non linear), and see how they perform on my model after the feature extraction task, using the MSRPC dataset. I tried 6 classifiers, among the most classical:
- Logistic Regression (Maximum Entropy)
- Linear Discriminant Analysis
- Classification and Regression Tree
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine

It turns out that the MaxEnt and the SVM algorithms perform best on my model with this dataset.
=======================================================

### SVM Classifier ###

This code implements the Support Vector Machine Classifier, visualize our set of vectors of features using some *pandas* tools, and through a grid-search enables to fid the best hyperparameters of the SVM Classifier in order to improve the performances of my decision-making algorithm. 
Some parameters have remained unchanged (like the *gamma coefficient*), and could be changed for better optimization of this model.
I perform a cross-fold validation (K=10) for test harness in order to avoid over-fitting.

