import pandas as pd
import numpy as np
import torch
from model import *

def load_test_data():
    # you can change the range to be 1 + the max of you training range to whatever number. 
    test_numbers = np.arange(90001, 100000)
    test_data = pd.DataFrame({"number": test_numbers})
    test_transformed = test_data['number'].apply(custom_transform)
    test_data = pd.concat([test_data, test_transformed], axis=1)
    return test_data

def test_model(learn, test_data):
    test_dl = learn.dls.test_dl(test_data)
    test_preds, _ = learn.get_preds(dl=test_dl)
    test_pred_class = test_preds.argmax(dim=1)
    test_actuals = test_data['is_prime']
    test_accuracy = (test_pred_class == torch.tensor(test_actuals)).float().mean()
    print(f"Test accuracy: {test_accuracy.item()*100:.2f}%")

def main():
    data = load_data()
    train_data, valid_data = split_data(data)
    learn = train_model(train_data)
    test_data = load_test_data()
    test_model(learn, test_data)

if __name__ == "__main__":
    main()
