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
def load_test_data():
    test_numbers = np.arange(1000001,1100001)
    test_data = pd.DataFrame({"number": test_numbers})
    test_transformed = test_data['number'].apply(custom_transform)
    test_data = pd.concat([test_data, test_transformed], axis=1)
    return test_data

def test_model(test_data):
    learn = load_learner('primenum.pkl')
    test_dl = learn.dls.test_dl(test_data)
    test_preds, _ = learn.get_preds(dl=test_dl)
    test_pred_class = test_preds.argmax(dim=1)
    test_actuals = test_data['is_prime']
    test_accuracy = (test_pred_class == torch.tensor(test_actuals)).float().mean()
    print(f"Test accuracy: {test_accuracy.item()*100:.2f}%")

def main():
    test_data = load_test_data()
    test_model(test_data)

if __name__ == "__main__":
    main()
