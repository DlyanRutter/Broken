from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import multiprocessing
import logging


logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameters = {
        'n_estimators': [10, 20, 30],
        'max_depth': [5, 10],
        'min_samples_split': [20, 50, 100],
        'learning_rate': [1.0],  # 0.1,0.5,
    }

    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching best hyperparameters on {} cores".format(njobs))

    clf = GridSearchCV(GradientBoostingClassifier(random_state=0),
                       param_grid=parameters,
                       cv=3,
                       n_jobs=njobs,
                       verbose=2,
                       )

    clf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {}".format(clf.best_params_))

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confuson matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm


def compute_slices(X, y, preds, feature):
    """
    function which compute the performance on slices of the categorical features
    ------
    X : np.array
        Data used for slices.
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized
    feature:
        feature on which to perform the slices
    Returns
    ------
    Dataframe with 
        precision : float
        recall : float
        fbeta : float
    for each of the unique values taken by the feature
    """    
    slice_options = X.loc[:,feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = X[feature]==option
        slice_y = y.iloc[slice_mask,:]
        slice_preds = preds.iloc[slice_mask,:]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, ['precision','recall','fbeta']] = precision, recall, fbeta

    return perf_df



def compute_slices(df, feature, encoder, y, preds):
    """
    function which compute the performance on slices of the categorical features
    ------
    X : np.array
        Data used for slices.
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized
    feature:
        feature on which to perform the slices
    Returns
    ------
    Dataframe with 
        precision : float
        recall : float
        fbeta : float
    for each of the unique values taken by the feature
    """    
    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature]==option
        #slice = df[df[feature]==option]
        #slice = encoder.transform(slice)

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    return perf_df