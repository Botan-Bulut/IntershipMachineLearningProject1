"""
Analysis of 1000 products.

Author: Botan Bulut

Date: 8/9/22
"""

# Imports

import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
# Display option

pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Data

data = pd.read_excel(r"C:\Users\botan\Desktop\RegisMachineLearningProject\data.xlsx")

# Categorical data description.

data_categorical_description = data.describe(include = "all")

data_categorical_description = data_categorical_description.transpose()

data_categorical_description.drop(columns = ["mean", "std", "min", "25%",
                                             "50%", "75%", "max"], inplace = True)

data_categorical_description = data_categorical_description.drop(index = ["Price", "Sales", "Revenue",
                                                                          "BSR", "FBA Fees", "Active Seller",
                                                                          "Ratings", "Review Count",
                                                                          "Images ", "Review velocity ",
                                                                          "Weight"])

data_categorical_description.to_excel(r"C:\Users\botan\Desktop\RegisMachineLearningProject\categoricalDataDescription.xlsx", index = True) # Exporting data


# Numerical Data description

data_numerical_description = data.describe()
data_numerical_description = data_numerical_description.transpose()
data_numerical_description.drop(columns = ["25%", "50%", "75%"], inplace = True)
data_numerical_description["sum"] = data[list(data_numerical_description.index)].sum()
data_numerical_description["var"] = data[list(data_numerical_description.index)].var()

data_numerical_description.to_excel(r"C:\Users\botan\Desktop\RegisMachineLearningProject\numericalDataDescription.xlsx", index = True) # Exporting data