# Dataset instructions

This project expects the original Kaggle Black Friday dataset files:

```text
train.csv
test.csv
```

Place them in this directory before running the notebook.

The repository does not store the raw CSV files because the dataset's redistribution rights should be respected and large data files do not belong in the source repository.

The analysis expects the standard Black Friday schema, including `User_ID`, `Product_ID`, `Gender`, `Age`, `Occupation`, `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`, `Product_Category_1`, `Product_Category_2`, `Product_Category_3`, and `Purchase` in the training data.

`Purchase` is the target column and is not present in the Kaggle test data.
