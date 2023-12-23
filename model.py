from fastai.tabular.all import *
import pandas as pd
import numpy as np
import torch


#isprime function, could possibly be replaced if you wanted to train a larger range of numbers 
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
#here are a few functions for features to train the model on, feel free to try and incorperate more and test    
def sum_of_digits(n):
    return sum(int(digit) for digit in str(n))

def num_divisors(n):
    divisors = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors += 2
    return divisors - 1 if n == int(n**0.5)**2 else divisors

def last_digit(n): return n % 10

def modulo_6(n): return n % 6

def prime_gap(n):
    def next_prime(num):
        def is_prime(k):
            return all(k % i for i in range(2, int(k**0.5) + 1)) and k > 1
        prime_candidate = num + 1
        while not is_prime(prime_candidate):
            prime_candidate += 1
        return prime_candidate
    return next_prime(n) - n


def custom_transform(x):
    return pd.Series({
        'sum_of_digits': sum_of_digits(x),
        'num_divisors': num_divisors(x),
        'last_digit': last_digit(x),
        'modulo_6': modulo_6(x),
        'is_prime': is_prime(x),
        'prime_gap': prime_gap(x)
    })
    
def load_data():
    # right here is the number range to train on
    numbers = np.arange(1, 1000000)
    data = pd.DataFrame({"number": numbers})
    data_transformed = data['number'].apply(custom_transform)
    data = pd.concat([data, data_transformed], axis=1)
    return data

def train_model(train_data):
    dls = TabularDataLoaders.from_df(train_data, path='.', y_names="is_prime",
                                 cat_names=None, cont_names=['sum_of_digits', 'num_divisors', 'last_digit', 'modulo_6', 'prime_gap'],
                                 procs=[Categorify, FillMissing, Normalize],
                                 valid_pct=0.2, seed=42, bs=64)
    learn = tabular_learner(dls, layers=[3000,2000,1000], metrics=accuracy)
    learn.fit_one_cycle(10)
    return learn

def main():
    data = load_data()
    learn = train_model(data)
    learn.export('C:/Users/alexh/Downloads/primenum.pkl')

if __name__ == "__main__":
    main()
