import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import joblib
import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score

from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer


class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

#
# Pipeline member to display the data at this stage of the transformation.
#
class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X


class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
# Activity,Sex,Age,Type,Species,Fatal
        self.mCategoricalPredictors = ["Activity", "Sex", "Type", "Species"]
        self.mNumericalPredictors = ["Age"]
        self.mLabels = ["Fatal"]

        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        # no fit necessary
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values


def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

# for validation set
def load_validation_data(filename):
    data = pd.read_csv(filename, index_col=0, dtype={"Activity": str, "Sex": str, "Type": str, "Species": str, "Fatal": str})
    return data

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file

def get_data(filename):
    logging.info(f"Loading data from {filename}")

    data = pd.read_csv(filename, index_col=0, dtype={ "Activity": str, "Sex": str, "Type": str, "Species": str, "Fatal": str })
    #logging.info(f"shape {data.shape}")
    return data

def load_data(my_args, filename):
    data = get_data(filename)
    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    y = data[label_column]


    return X, y

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label



def make_numerical_feature_pipeline(my_args):
    items = []
    
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))

    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Numerical Preprocessing")))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    ###
    ### sklearn's decision tree classifier requires all input features to be numerical
    ### one hot encoding accomplishes this.
    ###
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto')))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Categorial Preprocessing")))

    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline



#
#  TEST
#





def make_feature_pipeline(my_args):

    numerical_features = ["Age"] 
    numerical_transformer = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.impute.SimpleImputer(strategy='median')),('scaler', sklearn.preprocessing.StandardScaler())])
    categorical_features = ["Activity", "Sex", "Type", "Species"]
    categorical_transformer = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.impute.SimpleImputer(strategy='most_frequent')),('onehot', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),('cat', categorical_transformer, categorical_features)])

    return preprocessor

# I HAD TO USE SMOTE FOR UNBALANCED DATA
def make_full_pipeline_with_smote(my_args):
    preprocessor = make_feature_pipeline(my_args)
    smote = SMOTE(random_state=42)

    classifier = sklearn.ensemble.RandomForestClassifier(random_state=42)
    pipeline = ImblearnPipeline([('preprocessor', preprocessor),('smote', smote),('classifier', classifier)])
    return pipeline

###########################################

def do_fit_random_forest(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_full_pipeline_with_smote(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(pipeline, model_file)

    return model, pipeline
""" 
def do_fit_random_forest(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_random_forest_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return

 """

# do cross validation
def do_cross_validation_decision_tree(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_decision_tree_fit_pipeline(my_args)
    #scores = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=15, scoring="f1_micro")
    scores = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=sklearn.model_selection.LeaveOneOut(), scoring="f1_micro")
    print("Mean: {:6.4f} STD: {:6.4f}\nAll: {}".format(scores.mean(), scores.std(), scores))

    return



""" def do_cross_validation_random_forest(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_random_forest_pipeline(my_args)
    #scores = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=15, scoring="f1_micro")
    scores = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=sklearn.model_selection.LeaveOneOut(), scoring="f1_micro")
    print("Mean: {:6.4f} STD: {:6.4f}\nAll: {}".format(scores.mean(), scores.std(), scores))

    return """

def do_cross_validation_random_forest(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("Training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)


    pipeline = make_full_pipeline_with_smote(my_args)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_micro')

    print("Cross-validation scores:", scores)
    print("Mean score:", np.mean(scores))
    print("Standard deviation of scores:", np.std(scores))

""" 
def make_feature_pipeline(my_args):

    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline """

def make_random_forest_fit_pipeline(my_args):
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Final Preprocessing")))
   # items.append(("model", sklearn.ensemble.RandomForestClassifier(n_estimators=100))) 
    items.append(("model", sklearn.ensemble.RandomForestClassifier(n_estimators=my_args.n_estimators, 
                                                               bootstrap=my_args.bootstrap, 
                                                               oob_score=my_args.oob_score,
                                                               n_jobs=my_args.n_jobs,
                                                               max_samples=my_args.max_samples,
                                                               criterion=my_args.criterion,
                                                               max_depth=my_args.max_depth,
                                                               min_samples_split=my_args.min_samples_split,
                                                               min_samples_leaf=my_args.min_samples_split,
                                                               max_features=my_args.max_features,
                                                               max_leaf_nodes=my_args.max_leaf_nodes,
                                                               min_impurity_decrease=my_args.min_impurity_decrease,
                                                               random_state=my_args.random_state,
                                                               class_weight='balanced')))
    return sklearn.pipeline.Pipeline(items)

def make_decision_tree_fit_pipeline(my_args):
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Final Preprocessing")))
  # items.append(("model", sklearn.tree.DecisionTreeClassifier(max_depth=3, max_leaf_nodes=20)))
    items.append(("model", sklearn.tree.DecisionTreeClassifier(max_depth=my_args.max_depth, 
                                                               max_leaf_nodes=my_args.max_leaf_nodes, 
                                                               criterion=my_args.criterion,
                                                               splitter=my_args.splitter,
                                                               max_features=my_args.max_features,
                                                               min_samples_leaf=my_args.min_samples_leaf,
                                                               min_impurity_decrease=my_args.min_impurity_decrease,
                                                               class_weight='balanced')))

    return sklearn.pipeline.Pipeline(items)

""" def do_fit_random_forest(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_random_forest_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return """


def do_fit_random_forest(my_args, X_train, y_train):
    pipeline = make_random_forest_fit_pipeline(my_args)
    pipeline.fit(X_train, y_train)

    model_file = get_model_filename(my_args.model_file, my_args.train_file)
    joblib.dump(pipeline, model_file)
    return pipeline

""" def do_fit_decision_tree(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_decision_tree_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return model, pipeline """

# THIS IS FOR VALIDATION
def train_and_evaluate(my_args):
    X_train, y_train = load_data(my_args, my_args.train_file)
    X_test, y_test = load_data(my_args, my_args.test_file) 

    pipeline = make_full_pipeline_with_smote(my_args)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test) 
    print(classification_report(y_test, y_pred)) 



def do_fit_decision_tree(my_args, X_train, y_train):
    pipeline = make_decision_tree_fit_pipeline(my_args)
    pipeline.fit(X_train, y_train)

    model_file = get_model_filename(my_args.model_file, my_args.train_file)
    joblib.dump(pipeline, model_file)
    return pipeline




def do_random_search_dt(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_decision_tree_fit_pipeline(my_args)

    # use big params
    fit_params = make_decision_tree_params_grid_big(my_args)


    search_grid = sklearn.model_selection.RandomizedSearchCV(pipeline, fit_params,
                                                             scoring="f1_micro",
                                                             n_jobs=-1, verbose=1,
                                                             n_iter=my_args.n_search_iterations)
    search_grid.fit(X, y)
    
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    joblib.dump(search_grid, search_grid_file)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(search_grid.best_estimator_, model_file)

    return


def do_random_search_rf(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_random_forest_fit_pipeline(my_args)

    # use big params
    fit_params = make_random_forest_params_grid_big(my_args)


    search_grid = sklearn.model_selection.RandomizedSearchCV(pipeline, fit_params,
                                                             scoring="f1_micro",
                                                             n_jobs=-1, verbose=1,
                                                             n_iter=my_args.n_search_iterations)
    search_grid.fit(X, y)
    
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    joblib.dump(search_grid, search_grid_file)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(search_grid.best_estimator_, model_file)

    return


def get_feature_names(pipeline, X):
    primary_feature_names = list(X.columns[:])
    if 'polynomial-features' in pipeline['features'].named_steps:
        secondary_powers = pipeline['features']['polynomial-features'].powers_
        feature_names = []
        for powers in secondary_powers:
            s = ""
            for i in range(len(powers)):
                for j in range(powers[i]):
                    if len(s) > 0:
                        s += "*"
                    s += primary_feature_names[i]
            feature_names.append(s)
            logging.info("powers: {}  s: {}".format(powers, s))
    else:
        logging.info("polynomial-features not in features: {}".format(pipeline['features'].named_steps))
        feature_names = primary_feature_names
    return feature_names

def get_scale_offset(pipeline, count):
    if 'scaler' in pipeline['features'].named_steps:
        scaler = pipeline['features']['scaler']
        logging.info("scaler: {}".format(scaler))
        logging.info("scale: {}  mean: {}  var: {}".format(scaler.scale_, scaler.mean_, scaler.var_))
        theta_scale = 1.0 / scaler.scale_
        intercept_offset = scaler.mean_ / scaler.scale_
    else:
        theta_scale = np.ones(count)
        intercept_offset = np.zeros(count)
        logging.info("scaler not in features: {}".format(pipeline['features'].named_steps))
    return theta_scale, intercept_offset

def show_function(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))
    
    X, y = load_data(my_args, train_file)
    pipeline = joblib.load(model_file)

    feature_names = get_feature_names(pipeline, X)
    scale, offset = get_scale_offset(pipeline, len(feature_names))

    features = pipeline['features']
    X = features.transform(X)
    regressor = pipeline['model']

    intercept_offset = 0.0
    for i in range(len(regressor.coef_)):
        intercept_offset += regressor.coef_[i] * offset[i]

    s = "{}".format(regressor.intercept_[0]-intercept_offset)
    for i in range(len(regressor.coef_)):
        if len(feature_names[i]) > 0:
            t = "({}*{})".format(regressor.coef_[i]*scale[i], feature_names[i])
        else:
            t = "({})".format(regressor.coef_[i])
        if len(s) > 0:
            s += " + "
        s += t

    basename = get_basename(train_file)
    print("{}: {}".format(basename, s))
    return


def sklearn_metric(y, yhat):
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    table = "+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n|{:4d} |{:4d} |\n+-----+-----+\n".format(cm[0][0], cm[1][0], cm[0][1], cm[1][1])
    print(table)
    print()
    precision = sklearn.metrics.precision_score(y, yhat, pos_label='Y')
    recall = sklearn.metrics.recall_score(y, yhat, pos_label='Y')
    f1 = sklearn.metrics.f1_score(y, yhat, pos_label='Y')
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))
    return



def show_score(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    
    basename = get_basename(train_file)

    yhat_train = pipeline.predict(X_train)
    print()
    print("{}: train: ".format(basename))
    print()
    sklearn_metric(y_train, yhat_train)
    print()
    
    if my_args.show_test:
        yhat_test = pipeline.predict(X_test)
        print()
        print("{}: test: ".format(basename))
        print()
        sklearn_metric(y_test, yhat_test)
        print()
        
    return

def show_model(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    pipeline = joblib.load(model_file)
    tree = pipeline['model']

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(1, 1, 1)

    sklearn.tree.plot_tree(tree, ax=ax)
    fig.tight_layout()
    fig.savefig("tree.png", dpi=300)
    plt.close(fig)
    
    return

#
# SEARCH HYPERPARAMETERS
#


def make_numerical_predictor_params_big(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'median', 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 0, 1, 2, 3 ]

    return params

def make_categorical_predictor_params_big(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params_big(my_args):
    p1 = make_numerical_predictor_params_big(my_args)
    p2 = make_categorical_predictor_params_big(my_args)
    p1.update(p2)
    return p1

def make_decision_tree_params_grid_big(my_args):
    params = make_predictor_params_big(my_args)
    tree_params = {
        "model__criterion": [ "entropy", "gini" ],
        "model__splitter": [ "best", "random" ],
        "model__max_depth": [ 1, 2, 3, 4, None ],
        "model__min_samples_split": [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        "model__min_samples_leaf":  [ 0.01, 0.02, 0.04, 0.1 ],
        "model__max_features":  [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    params.update(tree_params)
    return params

def make_random_forest_params_grid_big(my_args):
    params = {
        "model__n_estimators": np.linspace(100, 1000, 10, dtype=int),
        "model__max_depth": [None, 10, 20, 30, 40, 50],
        "model__min_samples_split": np.linspace(2, 10, 5, dtype=int),
        "model__min_samples_leaf": np.linspace(1, 10, 5, dtype=int),
        "model__max_features": ['auto', 'sqrt', 'log2'],
        "model__bootstrap": [True, False],
        "model__max_leaf_nodes": [None, 10, 20, 30, 40, 50],
        "model__min_impurity_decrease": [0.0, 0.01, 0.1],
        "model__class_weight": [None, "balanced"]
    }
    return params

def show_best_params(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))


    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    if not os.path.exists(search_grid_file):
        raise Exception("Search grid file, '{}', does not exist.".format(search_grid_file))


    search_grid = joblib.load(search_grid_file)

    pp = pprint.PrettyPrinter(indent=4)
    print("Best Score:", search_grid.best_score_)
    print("Best Params:")
    pp.pprint(search_grid.best_params_)

    return

def get_search_grid_filename(search_grid_file, filename):
    if search_grid_file == "":
        basename = get_basename(filename)
        search_grid_file = "{}-search-grid.joblib".format(basename)
    return search_grid_file



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data With Linear Regression Using Pipeline')
    parser.add_argument('action', default='DT',
                        choices=["DT", "score", "show-model", "RF", "cross-validate-dt", "cross-validate-rf", "show-best-params", "random-search-dt", "random-search-rf"],
                        nargs='?', help="desired action")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--random-seed',   '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="Survived",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str,   help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--print-preprocessed-data', default=0,         type=int,   help="0 = don't do the debugging print, 1 = do print (default=0)")
    parser.add_argument('--search-grid-file', '-g', default="", type=str,   help="name of file for the search grid (default is constructed from train file name when fitting)")


    # decision tree parameters
    parser.add_argument('--criterion', default="entropy",  choices=('entropy', 'gini'),      help="use entropy or gini for impurity calculations")
    parser.add_argument('--splitter', default="best",      choices=('best', 'random'),       help="use best feature found or random feature to split")
    parser.add_argument('--max-features', default=None,    choices=('auto', 'sqrt', 'log2'), help="how many features to examine when looking for best splitter (default=None)")
    parser.add_argument('--max-depth', default=None,             type=int,   help="maximum depth of tree to learn (default=None)")
    parser.add_argument('--min-samples-split', default=2,        type=float,   help="minimum number of samples required to split a node.")
    parser.add_argument('--min-samples-leaf', default=1,         type=float,   help="minimum number of samples required to create a leaf node.")
    parser.add_argument('--max-leaf-nodes', default=None,        type=int,   help="maximum number of leaf nodes in learned tree (default=None)")
    parser.add_argument('--min-impurity-decrease', default=0.0,  type=float, help="minimum improvement in impurity to create a split (default=0.0)")

    # random forest parameters
    parser.add_argument('--n-estimators', default=100,         type=int,   help="Number of trees in the forest.")
    parser.add_argument('--bootstrap', default=True,         type=bool,   help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
    parser.add_argument('--oob_score', default=False,         type=bool,   help="Whether to use out-of-bag samples to estimate the generalization score. By default, accuracy_score is used. Provide a callable with signature metric(y_true, y_pred) to use a custom metric. Only available if bootstrap=True.")
    parser.add_argument('--n_jobs', default=None,         type=int,   help="The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.")
    parser.add_argument('--max-samples', default=None,        type=float,   help="If bootstrap is True, the number of samples to draw from X to train each base estimator.")
    parser.add_argument('--n-search-iterations', default=10,     type=int,   help="number of random iterations in randomized grid search.")
    parser.add_argument('--random-state', default=42,     type=int,   help="Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)")

    my_args = parser.parse_args(argv[1:])



    #
    # Do any special fixes/checks here
    #
    allowed_categorical_missing_strategies = ("most_frequent")
    if my_args.categorical_missing_strategy != "":
        if my_args.categorical_missing_strategy not in allowed_categorical_missing_strategies:
            raise Exception("Missing categorical strategy {} is not in the allowed list {}.".format(my_args.categorical_missing_strategy, allowed_categorical_missing_strategies))

    allowed_numerical_missing_strategies = ("mean", "median", "most_frequent")
    if my_args.numerical_missing_strategy != "":
        if my_args.numerical_missing_strategy not in allowed_numerical_missing_strategies:
            raise Exception("Missing numerical strategy {} is not in the allowed list {}.".format(my_args.numerical_missing_strategy, allowed_numerical_missing_strategies))

    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    # load training data
    X_train, y_train = load_data(my_args, my_args.train_file)
    # load val data
    X_val, y_val = load_data(my_args, "newAttacks-validate.csv")

    if my_args.action == 'DT':
        pipeline = do_fit_decision_tree(my_args, X_train, y_train)
        #eval validation data
        evaluate_model(pipeline, X_val, y_val, "validation")
    elif my_args.action == 'RF':
        pipeline = do_fit_random_forest(my_args, X_train, y_train)
        #eval validation data
        evaluate_model(pipeline, X_val, y_val, "validation")
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "show-model":
        show_model(my_args)
    elif my_args.action == 'cross-validate-dt':
        do_cross_validation_decision_tree(my_args)
    elif my_args.action == 'cross-validate-rf':
        do_cross_validation_random_forest(my_args)
    elif my_args.action == 'random-search-dt':
        do_random_search_dt(my_args)
    elif my_args.action == 'random-search-rf':
        do_random_search_rf(my_args)
    elif my_args.action == "show-best-params":
        show_best_params(my_args)
    else:
        raise Exception(f"Action: {my_args.action} is not known.")

def evaluate_model(pipeline, X, y, dataset_name):
    #evaluate the model
    y_pred = pipeline.predict(X)
    print(f"{dataset_name} Metrics:")
    print(sklearn.metrics.classification_report(y, y_pred))
    
if __name__ == "__main__":
    main(sys.argv)