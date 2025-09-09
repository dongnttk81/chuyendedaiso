import math
from typing import List, Tuple, Optional
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import seaborn as sns



def pca(x: np.ndarray, alpha: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    
    mu = np.mean(x, axis=0, keepdims=True)

    x_mu = x - mu

    cov_matrix = np.matmul(x_mu.T, x_mu) / x.shape[0]

    w, v = np.linalg.eig(cov_matrix)

    order = np.argsort(w)[::-1]

    w = w[order]
    v = v[:, order]

    rate = np.cumsum(w) / np.sum(w)

    r = np.where(rate >= alpha)

    U = v[:, :(r[0][0] + 1)]

    reduced_x = np.matmul(x, U)

    #print(reduced_x)

    return U, np.real(reduced_x) 


def construct_kernel(x: np.ndarray, type: str, sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> np.ndarray:

    if type == "linear":
        return np.matmul(x, x.T)
    elif type == "gaussian_rbf":
        assert sigma is not None
        dist_matrix = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        square_dist_matrix = np.sum(dist_matrix**2, axis=2)
        return np.exp(-square_dist_matrix/(2*sigma**2))
    elif type == "polynomial":
        assert r is not None and gamma is not None and d is not None
        return (r + gamma * np.matmul(x, x.T))**d
    elif type == "sigmoid":
        assert r is not None and gamma is not None
        return np.tanh(r + gamma * np.matmul(x, x.T))
    else:
        raise ValueError("{} is not a supported kernel type".format(type))
    
    
def kernel_pca(x: np.ndarray, alpha: float=0.95, type: str="gaussian_rbf", sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    if type == "linear":
        K = construct_kernel(x=x, type=type)
    elif type == "gaussian_rbf":
        K = construct_kernel(x=x, type=type, sigma=sigma)
    elif type == "polynomial":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma, d=d)
    elif type == "sigmoid":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma)
    else:
        raise ValueError("{} is not a supported kernel type".format(type))
    #print(K)
    n = K.shape[0]

    assert K.shape[0] == K.shape[1]

    K = np.matmul(np.eye(n) - np.ones(shape=(n, n))/n, K)
    K = np.matmul(K, np.eye(n) - np.ones(shape=(n, n))/n)

    eta, c = np.linalg.eig(K)

    eta = np.real(eta)
    c = np.real(c)

    order = np.argsort(eta)[::-1]
    eta = eta[order]
    c = c[:, order]

    lamb_da = eta / n

    c = c / (np.sqrt(eta + 1e-8)[np.newaxis, :])

    rate = np.cumsum(lamb_da) / np.sum(lamb_da)

    #print(rate)

    r = np.where(rate >= alpha)

    C = c[:, :(r[0][0] + 1)]

    reduced_data = np.matmul(C.T, K).T

    #print(reduced_data.shape)
    #print(reduced_data)

    return C, reduced_data


def preprocess_data(df: pd.DataFrame, remove_email_null: bool=True, use_text_categorical: bool=True, one_hot_encode: bool=True, return_df: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    columns_to_check = ['educ', 'wrkstat']
    df.dropna(subset=columns_to_check, inplace=True)
    df['emailtotal'] = np.where(df['emailhr'].isna() | df['emailmin'].isna(), np.nan, df['emailhr'] * 60 + df['emailmin'])
    df = df.drop(['emailhr', 'emailmin'], axis=1)

    df['educ'] = df['educ'].astype(np.int16)
        
    value_to_index = {}

    for col in df.columns.tolist():
        if col == 'emailtotal':
            continue
        try:
            unique_set = np.unique(df[col].to_numpy()).tolist()
            unique_set = [x for x in unique_set if not math.isnan(x)]
            unique_set.sort()
            value_to_index[col] = dict(zip(unique_set, unique_set))
        except:
            #print(col, list(set(df[col].tolist())))
            unique_set = list(set(df[col].tolist()))
            chosens = []
            for i in unique_set:
                if isinstance(i, float):
                    if math.isnan(i):
                        continue
                #print(type(i), i)
                chosens.append(i)
            unique_set = chosens
            unique_set.sort()

            if col == "advfront":
                value_to_index[col] = {'Strongly disagree': 1, 'Disagree': 2, 'Dont know': 3, 'Agree': 4, 'Strongly agree': 5}
            elif col == "polviews":
                value_to_index[col] = {'Extrmly conservative': 1, 'Conservative': 2, 'Slghtly conservative': 3, 'Moderate': 4, 'Slightly liberal': 5, 'Liberal': 6, 'Extremely liberal': 7}
            elif col == "educ" or col == "wrkstat":
                value_to_index[col] = dict(zip(unique_set, range(len(unique_set))))
            else:
                value_to_index[col] = dict(zip(unique_set, range(1, len(unique_set) + 1)))
                
    if not use_text_categorical:
        for col in df.columns.tolist():
            if col == 'emailtotal' or col == "wrkstat":
                continue
            df[col] = df[col].map(value_to_index[col])
            
    categorical_columns = ['harass5', 'polviews', 'advfront', 'snapchat', 'instagrm', 'wrkstat']
    
    if not use_text_categorical:
        df[categorical_columns] = df[categorical_columns].fillna(0)

        df['educ'] = df['educ'].fillna(21)
    else:
        df[categorical_columns] = df[categorical_columns].fillna("Unknown")

        df['educ'] = df['educ'].fillna("Unknown")
    

    if not use_text_categorical:
        casted_columns = ['harass5', 'educ', 'polviews', 'advfront', 'snapchat', 'instagrm']
    
        for col in casted_columns:
            df[col] = df[col].astype(np.int16)
            
            
    if one_hot_encode:
        categorical_columns = ['harass5', 'snapchat', 'instagrm', 'polviews', 'advfront', 'educ']
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    else:
        df_encoded = df.copy()
        
    if remove_email_null:
        df_encoded.dropna(subset=['emailtotal'], inplace=True)
        df_encoded['emailtotal'] = df_encoded['emailtotal'].astype(np.int16)
    else:
        df_encoded = df_encoded[df_encoded['emailtotal'].isna()]
        df_encoded = df_encoded.drop(['emailtotal'], axis=1)
    
    if remove_email_null:
        numerical_columns = ['emailtotal']
        scaler = StandardScaler()
        df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
        
    features_list = df_encoded.columns.tolist()
    features_list.remove('wrkstat')
    
    if return_df:
        return df_encoded, features_list, value_to_index
    
    y = df_encoded["wrkstat"]
    y = y.to_numpy()

    x = df_encoded.drop("wrkstat", axis=1).to_numpy()
    
    return x, y, features_list, value_to_index


# path = "./gss_16.rda"

# import pyreadr

# data = pyreadr.read_r(path)
# df = data["gss16"]

# x, y, _, _ = preprocess_data(df=df, remove_email_null=False, use_text_categorical=False)

# U, reduced_x = pca(x, alpha=0.45)

# print(reduced_x, reduced_x.shape)


def plot_bar_importance(importance):
    fig = plt.figure(figsize=(20, 20), facecolor='w')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.6)
    nums = len(importance)
    cols = 3
    rows = math.ceil(nums/cols)
    i = 0

    for cls in importance:
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title("Feature importance of class {}".format(cls))
        ax.bar(importance[cls].keys(), np.abs(list(importance[cls].values())))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.legend()
        ax.grid()
        i += 1
        
        
def plot_confusion_matrix(y_test, y_pred, classes, title="Confustion matrix"):
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    # Add legends for the heatmap
    bottom, top = plt.ylim()
    plt.ylim(bottom + 0.5, top - 0.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
    
    
def compute_aic(y_log_pred_prob: np.ndarray, p: int, sample_size: int=0):
    log_likelihood_elements = np.sum(y_log_pred_prob)
    if sample_size > 0:
        return -2 * log_likelihood_elements + np.log(sample_size) * p
    else:
        return -2 * log_likelihood_elements + 2 * p
    
    
def one_step_compute_aic(model, x_test, y_test, type="aic"):
    try:
        y_log_prob_pred = model.predict_log_proba(x_test)
    except:
        y_log_prob_pred = np.log(model.predict_proba(x_test) + 1e-8)
    classes = model.classes_
    pseudo_label_to_index = dict(zip(classes, range(len(classes))))
    y_test_encode = np.vectorize(lambda x: pseudo_label_to_index[x])(y_test)
    
    y_log_prob_pred = y_log_prob_pred[np.arange(len(y_log_prob_pred)), y_test_encode]
    
    if type == "aic":
        return compute_aic(y_log_pred_prob=y_log_prob_pred, p=x_test.shape[1], sample_size=0)
    else:
        return compute_aic(y_log_pred_prob=y_log_prob_pred, p=x_test.shape[1], sample_size=x_test.shape[0])