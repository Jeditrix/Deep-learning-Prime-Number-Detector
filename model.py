from fastai.tabular.all import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    numbers = np.arange(1, 90000)
    data = pd.DataFrame({"number": numbers})
    data_transformed = data['number'].apply(custom_transform)
    data = pd.concat([data, data_transformed], axis=1)
    return data

def split_data(data): return train_test_split(data, test_size=0.2, random_state=42)

def train_model(train_data):
    dls = TabularDataLoaders.from_df(train_data, path='.', y_names="is_prime",
                                     cat_names=None, cont_names=['sum_of_digits', 'num_divisors', 'last_digit', 'modulo_6'],
                                     procs=[Categorify, FillMissing, Normalize],
                                     valid_pct=0.2, seed=42, bs=64)
    learn = tabular_learner(dls, layers=[3000,2000,1000], metrics=accuracy)
    learn.fit_one_cycle(10)
    return learn

def main():
    data = load_data()
    train_data, valid_data = split_data(data)
    learn = train_model(train_data)
    print(f"Validation accuracy: {learn.recorder.values[-1][-1]*100:.2f}%")

if __name__ == "__main__":
    main()
