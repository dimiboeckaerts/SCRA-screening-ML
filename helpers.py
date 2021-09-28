# import libraries
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers import Dropout
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D


def plot_calibration_curve(X, y, model, model_name, outer_cv):
    """
    Plot calibration curve for model w/o and with calibration. 
    """

    # Calibrated with isotonic/sigmoid calibration
    isotonic = CalibratedClassifierCV(model, cv=outer_cv, method='isotonic')
    sigmoid = CalibratedClassifierCV(model, cv=outer_cv, method='sigmoid')
    lr = LogisticRegression(C=1.) # baseline

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    for clf, name in [(lr, 'Logistic'), (model, model_name), (isotonic, model_name + ' + Isotonic'), 
                      (sigmoid, model_name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label=name)
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    fig.savefig('calibration'+model_name+'.png', dpi=300)
    #plt.tight_layout()

def process_testdata(file):
    """
    Process the testdata from an .xlsx file into a transformed np.array that is ready for predictions.
    """
    # import libraries
    import re
    import pandas as pd
    from scipy import signal, integrate

    # get the data
    raw_data = pd.ExcelFile(file)
    sheetname = re.search('data/.+xlsx', file).group(0)[5:-5]
    test_data = np.matrix(raw_data.parse(sheetname, header=3)).T[1:,0:58] # first column (thus row in DF) is timestamp
    
    # get blanks and substract
    blanks = np.concatenate((test_data[34, :], test_data[35, :], test_data[46, :], test_data[47, :], test_data[58, :],
                             test_data[59, :], test_data[70, :], test_data[71, :]), axis=0)
    blanks_median = np.median(blanks, axis=0)
    test_data = np.delete(test_data, obj=[34, 35, 46, 47, 58, 59, 70, 71], axis=0)
    processed_data = np.asarray(test_data) #- blanks_median

    # transform the data
    transformed_sav1 = signal.savgol_filter(processed_data, window_length=11, polyorder=5, deriv=1)
    transformed_sav5 = signal.savgol_filter(processed_data, window_length=21, polyorder=2, deriv=2)
    transformed_int = np.zeros((processed_data.shape[0], processed_data.shape[1]))
    for i in range(transformed_int.shape[0]):
        transformed_int[i,:] = integrate.cumtrapz(processed_data[i,:], initial=0)
    
    return processed_data, transformed_sav1, transformed_sav5, transformed_int, blanks_median

def predict_testdata(trained_model, testdata, name):
    """
    Predict testcases on processed/transformed testdata for a given model.
    """
    import pandas as pd
    preds = trained_model.predict(testdata)
    probs = trained_model.predict_proba(testdata)
    probs = [probs[i][1] for i in range(probs.shape[0])]
    predictions_df = pd.DataFrame({name+'_probability': probs, name+'_prediction': preds})
    return predictions_df


def NestedGroupKFoldProba(model, X, y, parameter_grid, groups, n_classes, scorer, 
                     inner_cv=GroupKFold(n_splits=4), outer_cv=GroupKFold(n_splits=4), weights=[]):
    """
    Implements a nested version of GroupKFold cross-validation using GridSearchCV to evaluate models 
    that need hyperparameter tuning in settings where different groups exist in the available data.
    
    Dependencies: sklearn.model_selection, numpy
    
    Input:
    - X, y: features and labels (must be NumPy arrays).
    - model, parameter_grid: the model instance and its parameter grid to be optimized.
    - groups: the groups to use in both inner- and outer loop.
    - n_classes: the number of classes in the prediction problem
    - scorer: the scoring to use in inner loop.
    - inner_cv, outer_cv: the iterators for both CV-loops (default: GroupKFold(n_splits=4))
    - weights: sample weights to account for more important samples.
    
    Output: cross-validated predicted class probabilities
    """

    # define empty matrix to store performances (n CV runs and four performance metrics)
    probabilities = np.zeros((X.shape[0], n_classes))
    preds = np.zeros(X.shape[0])
    
    # define outer loop
    for train_outer, test_outer in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_outer], X[test_outer]
        y_train, y_test = y[train_outer], y[test_outer]
        groups_train, groups_test = groups[train_outer], groups[test_outer]
        
        # define inner loop (in GridSearchCV)
        tuned_model = GridSearchCV(model, cv=inner_cv, param_grid=parameter_grid, scoring=scorer)
        if len(weights) == 0:
            tuned_model.fit(X_train, y_train, groups=groups_train)
        else:
            weights_train = weights[train_outer]
            tuned_model.fit(X_train, y_train, groups=groups_train, **{'sample_weight': weights_train})
        
        # make predictions for test set (outer loop)
        y_probs = tuned_model.predict_proba(X_test)
        y_preds = tuned_model.predict(X_test)
        
        for i, index in enumerate(test_outer):
            probabilities[index,:] = y_probs[i,:]
            preds[index] = y_preds[i]
    
    return probabilities, preds


def NestedShuffledKFoldProba(model, X, y, parameter_grid, n_classes, scorer, 
                     inner_cv=StratifiedKFold(n_splits=10), outer_cv=StratifiedKFold(n_splits=10), 
                     weights=[]):
    """
    Implements a nested version of GroupKFold cross-validation using GridSearchCV to evaluate models 
    that need hyperparameter tuning in settings where different groups exist in the available data.
    
    Dependencies: sklearn.model_selection, numpy
    
    Input:
    - X, y: features and labels (must be NumPy arrays).
    - model, parameter_grid: the model instance and its parameter grid to be optimized.
    - n_classes: the number of classes in the prediction problem
    - scorer: the scoring to use in inner loop.
    - inner_cv, outer_cv: the StratifiedShuffleSplit iterators for both CV-loops (default n_splits: 10)
    - weights: sample weights to account for more important samples
    
    Output: cross-validated predicted class probabilities
    """

    # define empty matrix to store performances (n CV runs and four performance metrics)
    probabilities = np.zeros((X.shape[0], n_classes))
    preds = np.zeros(X.shape[0])
    
    # shuffle data
    indices = np.asarray(range(X.shape[0]), dtype=int)
    if len(weights) == 0:
        X, y, indices = shuffle(X, y, indices, random_state=0)
    else:
        X, y, weights, indices = shuffle(X, y, weights, indices, random_state=0)
    
    # define outer loop
    for train_outer, test_outer in outer_cv.split(X, y):
        X_train, X_test = X[train_outer], X[test_outer]
        y_train, y_test = y[train_outer], y[test_outer]
        indices_train, indices_test = indices[train_outer], indices[test_outer]
        
        # define inner loop (in GridSearchCV)
        tuned_model = GridSearchCV(model, cv=inner_cv, param_grid=parameter_grid, scoring=scorer)
        if len(weights) == 0:
            tuned_model.fit(X_train, y_train)
        else:
            weights_train = weights[train_outer]
            tuned_model.fit(X_train, y_train, **{'sample_weight': weights_train})
        
        # make predictions for test set (outer loop)
        y_probs = tuned_model.predict_proba(X_test)
        y_preds = tuned_model.predict(X_test)
        
        for i, index in enumerate(test_outer):
            original_index = indices_test[i]
            probabilities[original_index,:] = y_probs[i,:]
            preds[original_index] = y_preds[i]
    
    return probabilities, preds

#def simple_CNN(X_train, y_train, X_test, y_test):
#    """
#    A simple 1D convolutional neural network for time series classification.
#  
#    Dependencies: tensorflow, keras, numpy
#    """
#    # reshape data
#    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#    input_shape = (X_train.shape[1], 1)
#    output_shape = y_train.shape[1]
#    
#    # Define Sequential model with 3 layers
#    model = keras.Sequential([
#        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
#        MaxPooling1D(pool_size=2),
#        Conv1D(filters=64, kernel_size=2, activation='relu'),
#        ...
#        
#        Dense(output_shape, activation='softmax')
#        
#            layers.Dense(2, activation="relu", name="layer1"),
#            layers.Dense(3, activation="relu", name="layer2"),
#            layers.Dense(4, name="layer3")
#        ]
#    )
#    return
    
    