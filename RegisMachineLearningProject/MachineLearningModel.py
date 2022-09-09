
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import zscore


pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(suppress = True)
plt.rcParams['figure.dpi'] = 150
plt.rcParams["savefig.dpi"] = 150


data = pd.read_excel(r"C:\Users\botan\Desktop\RegisMachineLearningProject\data.xlsx")

desc_stat = pd.read_excel(r"C:\Users\botan\Desktop\RegisMachineLearningProject\numericalDataDescription.xlsx")
numerical_data_columns = ["Price", "Sales", "Revenue","BSR", "FBA Fees", 
                    "Active Seller","Ratings", "Review Count","Images ",
                    "Review velocity ","Weight", "Fulfillment"]
fulfillment_map = {"FBA" : 0, "MFN" : 1, "AMZ": 0}

data["Fulfillment"] = data["Fulfillment"].map(fulfillment_map)

pca_data = data[numerical_data_columns]
pca_data = pca_data.drop("Fulfillment", axis = 1)
pca_data = pca_data.dropna()

standardized_pca_data = zscore(pca_data)

scaled_data = preprocessing.scale(standardized_pca_data.T)

pca = PCA()
pca.fit(scaled_data)

pca_table = pca.transform(scaled_data)

pca_columns = ["PC" + str(i) for i in range(1, 12, 1)]

pca_dataframe = pd.DataFrame(index = pca_data.columns, columns = pca_columns, data = pca_table)

"""--- Scree Plot -- """

per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 3)
cum_per_var = per_var.cumsum()
plt.plot(pca_columns, per_var, color = "b", linewidth = 3, label = "% variance Explained")
plt.plot(pca_columns, cum_per_var, color = "k", linewidth = 3, marker = "o", label = "Cumulative % variance Explained")
plt.xlabel("Principal Component")
plt.ylabel("Variance (%)")
plt.legend()
plt.grid(True)
plt.ylim((0, 100))
plt.title("Cumulative Explained Variance")
plt.tight_layout()
plt.savefig(r"C:\Users\botan\Desktop\RegisMachineLearningProject\Screeplot.png")
plt.show()


columns = ["Revenue", "BSR", "Sales", "Review velocity ", "Price",
                    "Review Count","Category"]
data_of_interest = data[columns]

data_of_interest = data_of_interest.dropna()

X = data_of_interest[columns]
X = data_of_interest.drop("Category", axis = 1)
y = data_of_interest["Category"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)


joblib.dump(value = model, filename = r"C:\Users\botan\Desktop\RegisMachineLearningProject\DecisionTree.joblib") 
model = joblib.load(filename = r"C:\Users\botan\Desktop\RegisMachineLearningProject\DecisionTree.joblib") 

tree.export_graphviz(model, out_file = r"C:\Users\botan\Desktop\RegisMachineLearningProject\DecisionTree.dot",
                     feature_names = list(X.columns),
                     class_names = sorted(y.unique()),
                                          label = "all",
                                          rounded = True,
                                          filled = True)


"""---Output Scripting----"""

source_file = open(r"C:\Users\botan\Desktop\RegisMachineLearningProject\scriptOutput.txt", "w")
print("""
-- Regis Machine Learning Project--

Author: Botan Bulut
Date: 9/9/22
------------------------------------------------------------------------------------
""", file = source_file)

print(f"""

Imported dataset for this project:

{data}
      
""", file = source_file)

print(f"""
descriptive statistics for numerical data:

{desc_stat}

From Descriptive statistics, it can be easily observed that average and
variance values can manipulate analysis. Therefore, standardadization process
will be carried out.
------------------------------------------------------------------------------------
""", file = source_file)
print("""
To analyze numeric data columns, we need to ensure every element is numeric value.

""", file = source_file)
for i in numerical_data_columns: 
    
        print(f"{i} \t {is_numeric_dtype(data[i])}", file = source_file)
print("""
Table above shows every numeric column contains numeric data. Thus we can perform
statistical computation.
------------------------------------------------------------------------------------
""", file = source_file)
print(f"""
Standanrdized (zscore) dataset for PCA is given below:

{standardized_pca_data}

Now, PCA Analysis can be performed.
------------------------------------------------------------------------------------
""", file = source_file)

print(f"""
PCA Print is given below:

{pca_dataframe}  
    
""", file = source_file)
print(f"""
First 6 PC explain {cum_per_var[6]}% of the variance

""", file = source_file)

print("Loading scores for first 6 PC are given below: \n", file = source_file)

for i in range(1, 7, 1):
    
    dummy_variable = pd.Series(pca_dataframe["PC" + str(i)].abs().sort_values(ascending = False))
    print(f"Principal Component {i} Loading Scores:\n", file = source_file)
    print(f"{dummy_variable}\n",file = source_file)


print("""
------------------------------------------------------------------------------------

Variables that will be used as features are:

1 - Revenue
2 - Price
3 - Review Count
4 - Review velocity
5 - BSR
6 - Sales

These variables explains the variance in the dataset most.

------------------------------------------------------------------------------------

""", file = source_file)

print(f"""
Accuracy score of the test data = {accuracy_test}

END OF THE DOCUMENT
------------------------------------------------------------------------------------
""", file = source_file)

source_file.close()
