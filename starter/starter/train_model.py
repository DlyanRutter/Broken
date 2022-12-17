# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle, os

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slices
from ml.model import compute_confusion_matrix
import logging

logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
datapath = "../data/census.csv"
data = pd.read_csv(datapath)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split using stratify due to class imbalance
train, test = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
# Set train flag = False - We use the encoding from the train set
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# check if trained model already exists
savepath = '../model/trained_model.pkl'

# if saved model exits, load the model from disk
if os.path.isfile(savepath):
    
    model = pickle.load(open(savepath, 'rb'))

# Else Train and save a model.
else:
    model = train_model(X_train, y_train)
    # save model  to disk in ./model folder
    pickle.dump(model, open(savepath, 'wb'))
    logging.info(f"Model saved to disk: {savepath}")


# evaluate trained model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))

logging.info(f"Confusion matrix:\n{cm}")


# Compute performance on slices for categorical features
for feature in cat_features:
    performance_df = compute_slices(test, feature, encoder, y_test, preds)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)
