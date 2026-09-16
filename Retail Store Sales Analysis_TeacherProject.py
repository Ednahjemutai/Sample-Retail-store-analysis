#!/usr/bin/env python
# coding: utf-8

# # Retail Store Sales Analysis, Customer Purchase Prediction & Business Intelligence System
# 
# **Using Python, Machine Learning and Flask**
# 
# This notebook works through the Mini Capstone Project brief end-to-end:
# 
# 1. Data Inspection
# 2. Data Cleaning
# 3. Feature Engineering
# 4. Univariate Analysis
# 5. Bivariate Analysis
# 6. Multivariate Analysis
# 7. Exploratory Data Analysis (EDA) Summary
# 8. Statistical Analysis (T-Test, ANOVA, Chi-Square)
# 9. Machine Learning Problem Definition
# 10. Data Preprocessing
# 11. Machine Learning Model Development
# 12. Model Evaluation
# 13. Feature Importance
# 14. Model Saving
# 15. Flask Deployment (code + instructions)
# 16. Flask Deliverables & Folder Structure
# 17. Final Report — 15 Business Insights
# 
# > **Note:** Place `retail_store_sales.csv` in the same folder as this notebook (or update the path in the cell below) before running.

# In[1]:


# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

pd.set_option("display.max_columns", None)


# ---
# ## PART 1: DATA INSPECTION
# 
# **Objectives:** load the dataset, look at the first/last records, check dimensions, data types, missing values, duplicates, and descriptive statistics.

# In[2]:


# 1. Load the dataset
df = pd.read_csv("D:/datasets/retail_store_sales.csv")

df.head()


# In[3]:


# 2. First and last records
display(df.head())
display(df.tail())


# In[4]:


# 3. Dataset dimensions
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# In[5]:


# 5. Missing values
df.isnull().sum()


# In[6]:


# 6. Duplicate records
print("Number of duplicate rows:", df.duplicated().sum())


# In[9]:


# 7. Descriptive statistics -T : means transposing a column(swapping the rows and columns)
df.describe(include="all").T


# # Descriptive Statistics in Pandas (`.describe().T`)
# 
# ## Code
# 
# ```python
# df.describe(include="all").T
# ```
# 
# ## What does `.T` mean?
# 
# The **`.T`** stands for **transpose**.
# 
# A transpose swaps the **rows and columns** of a DataFrame, making the output easier to read.
# 
# ---
# 
# ## Without `.T`
# 
# When you run:
# 
# ```python
# df.describe(include="all")
# ```
# 
# You get descriptive statistics where the **statistics are rows** and the **dataset columns are columns**.
# 
# |       | Age | Salary | Score |
# |--------|-----|--------|-------|
# | count  | 100 | 100    | 100   |
# | mean   | 35  | 45000  | 78.5  |
# | std    | 8.2 | 12000  | 10.3  |
# | min    | 18  | 22000  | 45    |
# | 25%    | 29  | 36000  | 71    |
# | 50%    | 34  | 44000  | 79    |
# | 75%    | 41  | 52000  | 86    |
# | max    | 60  | 85000  | 98    |
# 
# ---
# 
# ## With `.T`
# 
# When you add `.T`:
# 
# ```python
# df.describe(include="all").T
# ```
# 
# The rows and columns are swapped.
# 
# |        | count | mean | std | min | 25% | 50% | 75% | max |
# |--------|-------|------|-----|-----|-----|-----|-----|-----|
# | Age    | 100   | 35   | 8.2 | 18  | 29  | 34  | 41  | 60  |
# | Salary | 100   | 45000|12000|22000|36000|44000|52000|85000|
# | Score  | 100   | 78.5 |10.3 |45   |71   |79   |86   |98   |
# 
# ---
# 
# ## Why use `.T`?
# 
# Using `.T` makes the descriptive statistics easier to interpret because:
# 
# - Each **row** represents one feature (column) in the dataset.
# - Each **column** represents a descriptive statistic.
# - It is especially useful when working with datasets that contain many variables.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Age": [20, 25, 30, 35],
#     "Score": [70, 80, 90, 100]
# })
# 
# # Descriptive statistics
# print(df.describe())
# 
# # Transposed descriptive statistics
# print(df.describe().T)
# ```
# 
# ### Output of `df.describe()`
# 
# ```text
#              Age      Score
# count   4.000000   4.000000
# mean   27.500000  85.000000
# std     6.454972  12.909944
# min    20.000000  70.000000
# 25%    23.750000  77.500000
# 50%    27.500000  85.000000
# 75%    31.250000  92.500000
# max    35.000000 100.000000
# ```
# 
# ### Output of `df.describe().T`
# 
# ```text
#        count  mean       std   min    25%    50%    75%    max
# Age      4.0  27.5   6.454972  20.0  23.75  27.50  31.25   35.0
# Score    4.0  85.0  12.909944  70.0  77.50  85.00  92.50  100.0
# ```
# 
# ---
# 
# ## Key Takeaways
# 
# - **`.T`** is a shortcut for **`.transpose()`**.
# - It swaps the rows and columns of a DataFrame.
# - It makes descriptive statistics easier to read and compare.
# - It is commonly used after `describe()` when analyzing datasets in Pandas.

# In[10]:


# Descriptive stats for the key numeric columns requested
df[["Price Per Unit", "Quantity", "Total Spent"]].describe()


# ### Answers — Part 1
# 
# 1. **Rows/Columns:** The dataset has **12,575 rows and 11 columns**.
# 2. **Categorical variables:** `Transaction ID`, `Customer ID`, `Category`, `Item`, `Payment Method`, `Location`, `Discount Applied` (boolean/categorical-like).
# 3. **Numerical variables:** `Price Per Unit`, `Quantity`, `Total Spent`. `Transaction Date` is a datetime once converted.
# 4. **Missing values:** Yes — several columns contain missing values.
# 5. **Columns with missing values:** `Item` (~1,213 missing), `Price Per Unit` (~609), `Quantity` (~604), `Total Spent` (~604), `Discount Applied` (~4,199 missing — these are transactions where no discount flag was recorded, effectively "No discount").
# 6. **Duplicates:** Run the duplicate check above — the raw file has no exact duplicate rows in this extract, but this should always be verified after each reload.
# 7. **Summary statistics:** Price Per Unit, Quantity, and Total Spent are summarised in the table above (mean, std, min, max, quartiles).

# ---
# ## PART 2: DATA CLEANING
# 
# **Objectives:** handle missing values, remove duplicates, correct inconsistent values, convert `Transaction Date` to datetime, and produce a clean dataset.

# In[11]:


# Work on a copy so the raw data stays untouched
clean_df = df.copy()


# In[13]:


# 1. Convert Transaction Date to datetime
clean_df["Transaction Date"] = pd.to_datetime(clean_df["Transaction Date"], errors="coerce")


# # Converting a Column to Date Format in Pandas
# 
# ## Code
# 
# ```python
# # Convert Transaction Date to datetime
# clean_df["Transaction Date"] = pd.to_datetime(
#     clean_df["Transaction Date"],
#     errors="coerce"
# )
# ```
# 
# ## Explanation
# 
# The `pd.to_datetime()` function converts a column containing dates stored as text (strings) into the **datetime** data type.
# 
# Converting a column to datetime allows you to perform date-based analysis, such as filtering records by date, extracting the year or month, sorting chronologically, and calculating time differences.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df["Transaction Date"]`
# 
# This selects the **Transaction Date** column from the DataFrame named `clean_df`.
# 
# ```python
# clean_df["Transaction Date"]
# ```
# 
# ---
# 
# ### `pd.to_datetime()`
# 
# This Pandas function converts values into a datetime format that Python and Pandas can recognize.
# 
# ```python
# pd.to_datetime(column_name)
# ```
# 
# ---
# 
# ### `errors="coerce"`
# 
# The `errors` parameter specifies what should happen if Pandas encounters an invalid date.
# 
# ```python
# errors="coerce"
# ```
# 
# When `errors="coerce"` is used:
# 
# - Valid dates are converted successfully.
# - Invalid or incorrectly formatted dates are replaced with **NaT (Not a Time)** instead of causing an error.
# - This allows data cleaning to continue without interrupting the program.
# 
# **Example**
# 
# | Original Value | Converted Value |
# |---------------|-----------------|
# | 2026-01-15 | 2026-01-15 |
# | 15/02/2026 | 2026-02-15 |
# | Invalid Date | NaT |
# | Missing Value | NaT |
# 
# ---
# 
# ## Why Convert Dates?
# 
# Converting date columns enables you to:
# 
# - Sort records by date.
# - Filter data by specific dates.
# - Extract the year, month, day, or weekday.
# - Calculate the number of days between dates.
# - Create time-series visualizations.
# - Perform trend and seasonal analysis.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Transaction Date": [
#         "2026-01-15",
#         "2026-02-20",
#         "Invalid Date",
#         "2026-04-10"
#     ]
# })
# 
# df["Transaction Date"] = pd.to_datetime(
#     df["Transaction Date"],
#     errors="coerce"
# )
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#   Transaction Date
# 0       2026-01-15
# 1       2026-02-20
# 2              NaT
# 3       2026-04-10
# ```
# 
# ---
# 
# ## Key Takeaways
# 
# - `pd.to_datetime()` converts text values into the **datetime** data type.
# - `errors="coerce"` replaces invalid dates with **NaT (Not a Time)** instead of raising an error.
# - Converting date columns is an essential data-cleaning step before performing time-based analysis.
# - Datetime columns enable filtering, sorting, grouping, and extracting useful date components such as year, month, and day.

# In[15]:


# 2. Discount Applied: missing means "no discount was applied" -> False
clean_df["Discount Applied"] = clean_df["Discount Applied"].fillna(False).astype(bool)


# # Handling Missing Values in a Boolean Column
# 
# ## Code
# 
# ```python
# # Discount Applied: missing means "no discount was applied" -> False
# clean_df["Discount Applied"] = clean_df["Discount Applied"].fillna(False).astype(bool)
# ```
# 
# ## Explanation
# 
# This code cleans the **Discount Applied** column by replacing missing values with **False** and converting the entire column into the **Boolean** (`bool`) data type.
# 
# In this dataset, a missing value means that **no discount was applied** during the transaction.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df["Discount Applied"]`
# 
# This selects the **Discount Applied** column from the DataFrame.
# 
# ```python
# clean_df["Discount Applied"]
# ```
# 
# ---
# 
# ### `.fillna(False)`
# 
# The `fillna()` function replaces missing (`NaN`) values with a specified value.
# 
# ```python
# .fillna(False)
# ```
# 
# Since a missing value indicates that no discount was given, it is replaced with **False**.
# 
# **Example**
# 
# | Before | After |
# |---------|-------|
# | True | True |
# | False | False |
# | NaN | False |
# | True | True |
# 
# ---
# 
# ### `.astype(bool)`
# 
# The `astype()` function converts the column into a Boolean data type.
# 
# ```python
# .astype(bool)
# ```
# 
# A Boolean column contains only two values:
# 
# - **True** – A discount was applied.
# - **False** – No discount was applied.
# 
# ---
# 
# ## Why Convert to Boolean?
# 
# Converting the column to Boolean:
# 
# - Makes the data more consistent.
# - Reduces memory usage compared to storing text values.
# - Makes filtering easier.
# - Simplifies calculations and summaries.
# 
# For example:
# 
# ```python
# # Count transactions with discounts
# clean_df["Discount Applied"].sum()
# ```
# 
# Since **True = 1** and **False = 0**, the sum gives the total number of discounted transactions.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# import numpy as np
# 
# df = pd.DataFrame({
#     "Discount Applied": [True, False, np.nan, True, np.nan]
# })
# 
# df["Discount Applied"] = (
#     df["Discount Applied"]
#     .fillna(False)
#     .astype(bool)
# )
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Discount Applied
# 0              True
# 1             False
# 2             False
# 3              True
# 4             False
# ```
# 
# ---
# 
# ## Key Takeaways
# 
# - `fillna(False)` replaces missing values with **False**.
# - `.astype(bool)` converts the column to the Boolean data type.
# - A Boolean column contains only **True** or **False** values.
# - Boolean columns are easier to analyze, filter, and summarize in Pandas.
# ```

# In[17]:


# 3. Item: fill missing item names with 'Unknown Item'
clean_df["Item"] = clean_df["Item"].fillna("Unknown Item")


# # Handling Missing Values in a Text Column
# 
# ## Code
# 
# ```python
# # Item: fill missing item names with 'Unknown Item'
# clean_df["Item"] = clean_df["Item"].fillna("Unknown Item")
# ```
# 
# ## Explanation
# 
# This code replaces missing values in the **Item** column with the text **"Unknown Item"**.
# 
# Instead of leaving missing values (`NaN`), a meaningful placeholder is used to indicate that the item name is unavailable.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df["Item"]`
# 
# This selects the **Item** column from the DataFrame.
# 
# ```python
# clean_df["Item"]
# ```
# 
# ---
# 
# ### `.fillna("Unknown Item")`
# 
# The `fillna()` function replaces all missing (`NaN`) values with the specified value.
# 
# ```python
# .fillna("Unknown Item")
# ```
# 
# In this case:
# 
# - Existing item names remain unchanged.
# - Missing values are replaced with **"Unknown Item"**.
# 
# ---
# 
# ## Example
# 
# ### Before Cleaning
# 
# | Item |
# |------|
# | Laptop |
# | Phone |
# | NaN |
# | Printer |
# | NaN |
# 
# ### After Cleaning
# 
# | Item |
# |------|
# | Laptop |
# | Phone |
# | Unknown Item |
# | Printer |
# | Unknown Item |
# 
# ---
# 
# ## Why Replace Missing Text Values?
# 
# Replacing missing values with a descriptive label:
# 
# - Prevents missing values from causing errors during analysis.
# - Makes reports and dashboards easier to understand.
# - Preserves all records instead of deleting rows.
# - Clearly indicates that the item information is unavailable.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# import numpy as np
# 
# df = pd.DataFrame({
#     "Item": ["Laptop", "Phone", np.nan, "Printer", np.nan]
# })
# 
# df["Item"] = df["Item"].fillna("Unknown Item")
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#            Item
# 0       Laptop
# 1        Phone
# 2  Unknown Item
# 3      Printer
# 4  Unknown Item
# ```
# 
# ---
# 
# ## Key Takeaways
# 
# - `fillna()` replaces missing (`NaN`) values with a specified value.
# - `"Unknown Item"` is used as a placeholder for missing item names.
# - Using descriptive placeholders improves data quality and makes reports easier to interpret.
# - This approach preserves all records while clearly identifying missing information.

# In[18]:


# 4. Price Per Unit: fill missing values with the median price for that Category
clean_df["Price Per Unit"] = clean_df.groupby("Category")["Price Per Unit"].transform(
    lambda x: x.fillna(x.median())
)


# # Filling Missing Values with the Median of Each Category
# 
# ## Code
# 
# ```python
# # Price Per Unit: fill missing values with the median price for that Category
# clean_df["Price Per Unit"] = clean_df.groupby("Category")["Price Per Unit"].transform(
#     lambda x: x.fillna(x.median())
# )
# ```
# 
# ## Explanation
# 
# This code fills missing values in the **Price Per Unit** column using the **median price** of products within the same **Category**.
# 
# Instead of using a single value for the entire dataset, each category receives its own median value, making the imputation more accurate and meaningful.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df.groupby("Category")`
# 
# The `groupby()` function divides the dataset into groups based on the **Category** column.
# 
# ```python
# clean_df.groupby("Category")
# ```
# 
# For example:
# 
# | Category | Price Per Unit |
# |----------|----------------|
# | Electronics | 1500 |
# | Electronics | NaN |
# | Furniture | 800 |
# | Furniture | NaN |
# 
# The data is grouped into:
# 
# - Electronics
# - Furniture
# 
# Each group is processed independently.
# 
# ---
# 
# ### `["Price Per Unit"]`
# 
# This selects the **Price Per Unit** column from each category.
# 
# ```python
# ["Price Per Unit"]
# ```
# 
# ---
# 
# ### `.transform()`
# 
# The `transform()` function performs an operation on each group and returns a result with the **same number of rows** as the original DataFrame.
# 
# ```python
# .transform(...)
# ```
# 
# Unlike `groupby().agg()`, which returns one value per group, `transform()` returns a value for every original row, making it ideal for replacing missing values.
# 
# ---
# 
# ### `lambda x: x.fillna(x.median())`
# 
# A **lambda function** is an anonymous (unnamed) function used for short operations.
# 
# ```python
# lambda x: x.fillna(x.median())
# ```
# 
# For each category:
# 
# 1. Calculate the **median** price.
# 2. Replace any missing (`NaN`) values with that median.
# 
# For example:
# 
# **Electronics**
# 
# | Price |
# |-------|
# | 1500 |
# | 1800 |
# | NaN |
# | 2000 |
# 
# Median = **1800**
# 
# After filling:
# 
# | Price |
# |-------|
# | 1500 |
# | 1800 |
# | 1800 |
# | 2000 |
# 
# ---
# 
# ## Why Use the Median Instead of the Mean?
# 
# The **median** is often preferred because it is **less affected by outliers**.
# 
# Example:
# 
# Prices:
# 
# ```text
# 100, 120, 150, 160, 5000
# ```
# 
# - Mean = 1106
# - Median = 150
# 
# The median better represents the typical price when extreme values exist.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# import numpy as np
# 
# df = pd.DataFrame({
#     "Category": ["Electronics", "Electronics", "Electronics",
#                  "Furniture", "Furniture"],
#     "Price Per Unit": [1500, np.nan, 2000, 800, np.nan]
# })
# 
# df["Price Per Unit"] = (
#     df.groupby("Category")["Price Per Unit"]
#       .transform(lambda x: x.fillna(x.median()))
# )
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#        Category  Price Per Unit
# 0  Electronics          1500.0
# 1  Electronics          1750.0
# 2  Electronics          2000.0
# 3     Furniture           800.0
# 4     Furniture           800.0
# ```
# 
# **Explanation of the output:**
# 
# - **Electronics** prices are **1500** and **2000**.
#   - Median = **1750**
#   - The missing value is replaced with **1750**.
# - **Furniture** has only one available price (**800**).
#   - Median = **800**
#   - The missing value is replaced with **800**.
# 
# ---
# 
# ## Why Use `groupby()` and `transform()`?
# 
# Using category-specific medians:
# 
# - Produces more realistic values than using one median for the entire dataset.
# - Preserves differences between product categories.
# - Improves the quality of statistical analysis and machine learning models.
# 
# ---
# 
# ## Key Takeaways
# 
# - `groupby("Category")` groups records by product category.
# - `transform()` returns a result for every row, making it suitable for filling missing values.
# - `lambda x: x.fillna(x.median())` replaces missing prices with the **median** price of that category.
# - The **median** is preferred over the mean because it is less influenced by extreme values (outliers).
# - Group-based imputation helps preserve the characteristics of each category and improves data quality.

# In[19]:


# 5. Quantity: fill missing values with the overall median quantity
clean_df["Quantity"] = clean_df["Quantity"].fillna(clean_df["Quantity"].median())


# # Filling Missing Values with the Overall Median
# 
# ## Code
# 
# ```python
# # Quantity: fill missing values with the overall median quantity
# clean_df["Quantity"] = clean_df["Quantity"].fillna(clean_df["Quantity"].median())
# ```
# 
# ## Explanation
# 
# This code replaces missing values in the **Quantity** column with the **overall median quantity** calculated from the entire dataset.
# 
# Using the median ensures that missing values are replaced with a value that represents the typical quantity while minimizing the effect of extreme values (outliers).
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df["Quantity"]`
# 
# This selects the **Quantity** column from the DataFrame.
# 
# ```python
# clean_df["Quantity"]
# ```
# 
# ---
# 
# ### `.median()`
# 
# The `median()` function calculates the **middle value** of a numeric column after sorting the values in ascending order.
# 
# ```python
# clean_df["Quantity"].median()
# ```
# 
# For example:
# 
# ```text
# 2, 3, 4, 5, 8
# ```
# 
# Median = **4**
# 
# If there is an even number of values, the median is the average of the two middle values.
# 
# Example:
# 
# ```text
# 2, 4, 6, 8
# ```
# 
# Median = **(4 + 6) / 2 = 5**
# 
# ---
# 
# ### `.fillna()`
# 
# The `fillna()` function replaces all missing (`NaN`) values with the specified value.
# 
# ```python
# .fillna(clean_df["Quantity"].median())
# ```
# 
# In this case:
# 
# 1. Calculate the overall median quantity.
# 2. Replace every missing value with that median.
# 
# ---
# 
# ## Example
# 
# ### Before Cleaning
# 
# | Quantity |
# |----------|
# | 2 |
# | 5 |
# | NaN |
# | 8 |
# | 4 |
# 
# Median = **4.5**
# 
# ### After Cleaning
# 
# | Quantity |
# |----------|
# | 2 |
# | 5 |
# | 4.5 |
# | 8 |
# | 4 |
# 
# ---
# 
# ## Complete Example
# 
# ```python
# import pandas as pd
# import numpy as np
# 
# df = pd.DataFrame({
#     "Quantity": [2, 5, np.nan, 8, 4]
# })
# 
# df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Quantity
# 0       2.0
# 1       5.0
# 2       4.5
# 3       8.0
# 4       4.0
# ```
# 
# ---
# 
# ## Why Use the Median?
# 
# The median is preferred because it is **less affected by outliers** than the mean.
# 
# For example:
# 
# ```text
# 1, 2, 3, 4, 100
# ```
# 
# - Mean = **22**
# - Median = **3**
# 
# The median provides a more representative value when the data contains unusually large or small values.
# 
# ---
# 
# ## Why Use the Overall Median?
# 
# Unlike group-based imputation, this approach uses **one median value for the entire dataset**.
# 
# It is appropriate when:
# 
# - The dataset has no meaningful groups (such as categories).
# - Quantities are expected to have a similar distribution across all records.
# - A simple and reliable method is sufficient for handling missing values.
# 
# ---
# 
# ## Key Takeaways
# 
# - `median()` calculates the middle value of a numeric column.
# - `fillna()` replaces missing (`NaN`) values with a specified value.
# - Using the **overall median** provides a simple and robust method for imputing missing numeric values.
# - The median is less sensitive to outliers than the mean, making it a preferred choice for many data-cleaning tasks.

# In[20]:


# 6. Total Spent: recompute from Price Per Unit * Quantity to fix missing/inconsistent totals
clean_df["Total Spent"] = (clean_df["Price Per Unit"] * clean_df["Quantity"]).round(2)


# # Recalculating the Total Spent Column
# 
# ## Code
# 
# ```python
# # Total Spent: recompute from Price Per Unit * Quantity to fix missing/inconsistent totals
# clean_df["Total Spent"] = (
#     clean_df["Price Per Unit"] * clean_df["Quantity"]
# ).round(2)
# ```
# 
# ## Explanation
# 
# This code recalculates the **Total Spent** for every transaction by multiplying the **Price Per Unit** by the **Quantity** purchased.
# 
# Recomputing the values ensures that the **Total Spent** column is accurate and consistent, especially if it contains missing values or incorrect calculations.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df["Price Per Unit"]`
# 
# This selects the **Price Per Unit** column.
# 
# ```python
# clean_df["Price Per Unit"]
# ```
# 
# This column contains the cost of one unit of an item.
# 
# Example:
# 
# | Price Per Unit |
# |---------------:|
# | 150.00 |
# | 250.50 |
# | 80.75 |
# 
# ---
# 
# ### `clean_df["Quantity"]`
# 
# This selects the **Quantity** column.
# 
# ```python
# clean_df["Quantity"]
# ```
# 
# This column contains the number of units purchased.
# 
# Example:
# 
# | Quantity |
# |---------:|
# | 2 |
# | 5 |
# | 3 |
# 
# ---
# 
# ### Multiplication (`*`)
# 
# The multiplication operator calculates the total amount spent for each transaction.
# 
# ```python
# clean_df["Price Per Unit"] * clean_df["Quantity"]
# ```
# 
# Formula:
# 
# ```text
# Total Spent = Price Per Unit × Quantity
# ```
# 
# Example:
# 
# | Price Per Unit | Quantity | Total Spent |
# |---------------:|---------:|------------:|
# | 150.00 | 2 | 300.00 |
# | 250.50 | 5 | 1252.50 |
# | 80.75 | 3 | 242.25 |
# 
# ---
# 
# ### `.round(2)`
# 
# The `round()` function rounds the calculated values to a specified number of decimal places.
# 
# ```python
# .round(2)
# ```
# 
# Using `2` means the values are rounded to **two decimal places**, which is the standard format for monetary values.
# 
# Example:
# 
# | Before | After |
# |--------:|------:|
# | 1252.49999 | 1252.50 |
# | 242.2567 | 242.26 |
# | 300.0 | 300.00 |
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Price Per Unit": [150.00, 250.50, 80.75],
#     "Quantity": [2, 5, 3]
# })
# 
# df["Total Spent"] = (
#     df["Price Per Unit"] * df["Quantity"]
# ).round(2)
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Price Per Unit  Quantity  Total Spent
# 0          150.00         2       300.00
# 1          250.50         5      1252.50
# 2           80.75         3       242.25
# ```
# 
# ---
# 
# ## Why Recalculate the Total Spent?
# 
# Recomputing the **Total Spent** column helps to:
# 
# - Correct incorrect totals caused by data entry errors.
# - Replace missing values with accurate calculations.
# - Ensure consistency between **Price Per Unit**, **Quantity**, and **Total Spent**.
# - Improve the reliability of financial reports and data analysis.
# 
# ---
# 
# ## Key Takeaways
# 
# - The **Total Spent** is calculated using the formula:
# 
#   ```text
#   Total Spent = Price Per Unit × Quantity
#   ```
# 
# - Multiplying the two columns produces an accurate total for every transaction.
# - `.round(2)` formats the results to two decimal places, making them suitable for currency values.
# - Recalculating totals is an important data-cleaning step that improves data accuracy and consistency before analysis.

# In[21]:


# 7. Remove exact duplicate rows
before = clean_df.shape[0]
clean_df = clean_df.drop_duplicates()
after = clean_df.shape[0]
print(f"Duplicates removed: {before - after}")


# # Removing Exact Duplicate Rows
# 
# ## Code
# 
# ```python
# # Remove exact duplicate rows
# before = clean_df.shape[0]
# 
# clean_df = clean_df.drop_duplicates()
# 
# after = clean_df.shape[0]
# 
# print(f"Duplicates removed: {before - after}")
# ```
# 
# ## Explanation
# 
# This code removes **exact duplicate rows** from the dataset and reports how many duplicates were removed.
# 
# Duplicate records can occur due to data entry errors, repeated imports, or system issues. Removing them improves data quality and prevents inaccurate analysis.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df.shape[0]`
# 
# The `shape` attribute returns the dimensions of a DataFrame.
# 
# ```python
# clean_df.shape
# ```
# 
# Example output:
# 
# ```text
# (1000, 8)
# ```
# 
# This means:
# 
# - **1000 rows**
# - **8 columns**
# 
# To obtain only the number of rows:
# 
# ```python
# clean_df.shape[0]
# ```
# 
# Example:
# 
# ```python
# before = clean_df.shape[0]
# ```
# 
# This stores the number of rows **before** duplicates are removed.
# 
# ---
# 
# ### `drop_duplicates()`
# 
# The `drop_duplicates()` function removes rows that are **exactly identical** across all columns.
# 
# ```python
# clean_df = clean_df.drop_duplicates()
# ```
# 
# By default:
# 
# - The **first occurrence** of each duplicate row is kept.
# - Any additional identical rows are removed.
# 
# ---
# 
# ### Example
# 
# #### Before Removing Duplicates
# 
# | Transaction ID | Item | Quantity |
# |---------------|------|----------|
# | T001 | Laptop | 2 |
# | T002 | Phone | 1 |
# | T002 | Phone | 1 |
# | T003 | Printer | 4 |
# 
# #### After Removing Duplicates
# 
# | Transaction ID | Item | Quantity |
# |---------------|------|----------|
# | T001 | Laptop | 2 |
# | T002 | Phone | 1 |
# | T003 | Printer | 4 |
# 
# The duplicate record has been removed.
# 
# ---
# 
# ### Count Rows After Cleaning
# 
# ```python
# after = clean_df.shape[0]
# ```
# 
# This stores the number of rows **after** duplicate removal.
# 
# ---
# 
# ### Calculate the Number of Duplicates Removed
# 
# ```python
# before - after
# ```
# 
# For example:
# 
# ```text
# Rows before = 1000
# Rows after = 985
# 
# Duplicates removed = 15
# ```
# 
# ---
# 
# ### Print the Result
# 
# ```python
# print(f"Duplicates removed: {before - after}")
# ```
# 
# This uses an **f-string (formatted string)** to display the number of duplicate rows removed.
# 
# Example output:
# 
# ```text
# Duplicates removed: 15
# ```
# 
# ---
# 
# ## Complete Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "ID": [1, 2, 2, 3],
#     "Item": ["Laptop", "Phone", "Phone", "Printer"]
# })
# 
# before = df.shape[0]
# 
# df = df.drop_duplicates()
# 
# after = df.shape[0]
# 
# print(f"Duplicates removed: {before - after}")
# ```
# 
# ### Output
# 
# ```text
# Duplicates removed: 1
# ```
# 
# Resulting DataFrame:
# 
# | ID | Item |
# |---:|------|
# | 1 | Laptop |
# | 2 | Phone |
# | 3 | Printer |
# 
# ---
# 
# ## Why Remove Duplicate Rows?
# 
# Removing duplicate records helps to:
# 
# - Improve data quality.
# - Prevent double-counting during analysis.
# - Produce more accurate statistics and visualizations.
# - Reduce unnecessary storage space.
# - Improve the performance of data processing tasks.
# 
# ---
# 
# ## Key Takeaways
# 
# - `shape[0]` returns the number of rows in a DataFrame.
# - `drop_duplicates()` removes rows that are identical across all columns.
# - Storing the row count before and after cleaning allows you to measure how many duplicates were removed.
# - Using an **f-string** makes it easy to display the result in a readable format.
# - Removing duplicates is an essential data-cleaning step before performing analysis or building machine learning models.

# In[22]:


# 8. Drop rows where the Transaction Date could not be parsed
clean_df = clean_df.dropna(subset=["Transaction Date"])

print("Remaining missing values:\n", clean_df.isnull().sum())
print("\nFinal clean shape:", clean_df.shape)


# # Removing Rows with Invalid Transaction Dates
# 
# ## Code
# 
# ```python
# # Drop rows where the Transaction Date could not be parsed
# clean_df = clean_df.dropna(subset=["Transaction Date"])
# 
# print("Remaining missing values:\n", clean_df.isnull().sum())
# print("\nFinal clean shape:", clean_df.shape)
# ```
# 
# ## Explanation
# 
# This code performs two important data-cleaning tasks:
# 
# 1. Removes rows where the **Transaction Date** is missing or invalid.
# 2. Checks the dataset for any remaining missing values and displays the final size of the cleaned dataset.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `dropna()`
# 
# The `dropna()` function removes rows or columns that contain missing values (`NaN` or `NaT`).
# 
# ```python
# clean_df.dropna()
# ```
# 
# By default, it removes any row containing at least one missing value.
# 
# ---
# 
# ### `subset=["Transaction Date"]`
# 
# The `subset` parameter tells Pandas to check only the specified column.
# 
# ```python
# subset=["Transaction Date"]
# ```
# 
# This means:
# 
# - Remove rows where **Transaction Date** is missing (`NaT`).
# - Keep rows even if other columns still contain missing values.
# 
# Earlier in the cleaning process, invalid dates were converted to **NaT** using:
# 
# ```python
# pd.to_datetime(..., errors="coerce")
# ```
# 
# This step removes those invalid records.
# 
# ---
# 
# ## Example
# 
# ### Before Cleaning
# 
# | Transaction Date | Item | Quantity |
# |-----------------|------|----------|
# | 2026-01-10 | Laptop | 2 |
# | NaT | Phone | 1 |
# | 2026-03-05 | Printer | 4 |
# | NaT | Mouse | 3 |
# 
# ### After Cleaning
# 
# | Transaction Date | Item | Quantity |
# |-----------------|------|----------|
# | 2026-01-10 | Laptop | 2 |
# | 2026-03-05 | Printer | 4 |
# 
# Rows with invalid or missing dates have been removed.
# 
# ---
# 
# ### Check Remaining Missing Values
# 
# ```python
# clean_df.isnull().sum()
# ```
# 
# #### `isnull()`
# 
# The `isnull()` function checks every cell in the DataFrame.
# 
# - Missing values are marked as **True**.
# - Existing values are marked as **False**.
# 
# #### `sum()`
# 
# The `sum()` function counts the number of **True** values in each column.
# 
# Example output:
# 
# ```text
# Transaction Date    0
# Customer Name       0
# Item                0
# Category            0
# Price Per Unit      0
# Quantity            0
# Total Spent         0
# Discount Applied    0
# dtype: int64
# ```
# 
# A value of **0** means there are no missing values remaining in that column.
# 
# ---
# 
# ### Display the Final Dataset Shape
# 
# ```python
# clean_df.shape
# ```
# 
# The `shape` attribute returns:
# 
# ```text
# (Number of Rows, Number of Columns)
# ```
# 
# Example:
# 
# ```text
# (985, 8)
# ```
# 
# This means the cleaned dataset contains:
# 
# - **985 rows**
# - **8 columns**
# 
# ---
# 
# ### Print the Results
# 
# ```python
# print("Remaining missing values:\n", clean_df.isnull().sum())
# ```
# 
# Displays the number of missing values remaining in each column.
# 
# ```python
# print("\nFinal clean shape:", clean_df.shape)
# ```
# 
# Displays the dimensions of the cleaned dataset.
# 
# Example output:
# 
# ```text
# Remaining missing values:
# 
# Transaction Date    0
# Customer Name       0
# Item                0
# Category            0
# Price Per Unit      0
# Quantity            0
# Total Spent         0
# Discount Applied    0
# dtype: int64
# 
# Final clean shape: (985, 8)
# ```
# 
# ---
# 
# ## Why Remove Rows with Invalid Dates?
# 
# Dates are often essential for:
# 
# - Time-series analysis.
# - Monthly or yearly sales reports.
# - Trend analysis.
# - Sales forecasting.
# - Data visualization over time.
# 
# Keeping records with invalid dates can lead to inaccurate results or errors during analysis.
# 
# ---
# 
# ## Key Takeaways
# 
# - `dropna(subset=["Transaction Date"])` removes only rows with missing or invalid **Transaction Date** values.
# - `isnull().sum()` counts the remaining missing values in each column.
# - `shape` returns the dimensions of the cleaned DataFrame.
# - Printing these results helps verify that the data-cleaning process was successful.
# - Removing records with invalid dates ensures the dataset is reliable for time-based analysis and reporting.

# ### Answers — Part 2
# 
# 1. **How were missing values handled?**
#    - `Discount Applied` missing → treated as **False** (no discount recorded).
#    - `Item` missing → labeled `"Unknown Item"` so the row is kept but flagged.
#    - `Price Per Unit` missing → filled with the **median price for that product Category** (more accurate than a global median).
#    - `Quantity` missing → filled with the **overall median quantity**.
#    - `Total Spent` → **recomputed** as `Price Per Unit * Quantity` after the two inputs were cleaned, instead of trusting the original (possibly inconsistent) totals.
# 2. **Were duplicates removed?** Yes, `drop_duplicates()` was applied and the count of removed rows is printed above.
# 3. **Data quality issues identified:** missing prices/quantities/totals, missing item labels, missing discount flags, and `Transaction Date` stored as text instead of a date type.
# 4. **Transformations performed:** datetime conversion, boolean conversion of the discount flag, group-wise imputation of price, median imputation of quantity, recomputation of total spend, and duplicate removal.

# ---
# ## PART 3: FEATURE ENGINEERING
# 
# Create the business features requested in the brief: date features, spending category, quantity category, discount status, and revenue band.

# In[24]:


fe_df = clean_df.copy()


# # Creating a Copy of the Cleaned Dataset
# 
# ## Code
# 
# ```python
# fe_df = clean_df.copy()
# ```
# 
# ## Explanation
# 
# This code creates a **copy** of the cleaned DataFrame (`clean_df`) and stores it in a new DataFrame called `fe_df`.
# 
# The abbreviation **`fe_df`** stands for **Feature Engineering DataFrame**.
# 
# Creating a copy allows you to perform **feature engineering** (creating new variables or modifying existing ones) without changing the original cleaned dataset.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `clean_df`
# 
# This is the cleaned DataFrame that has already undergone data cleaning operations such as:
# 
# - Handling missing values.
# - Removing duplicate records.
# - Correcting data types.
# - Recalculating totals.
# - Removing invalid records.
# 
# Example:
# 
# ```python
# clean_df
# ```
# 
# ---
# 
# ### `.copy()`
# 
# The `.copy()` method creates a completely independent copy of a DataFrame.
# 
# ```python
# clean_df.copy()
# ```
# 
# This means that any changes made to the new DataFrame **do not affect** the original DataFrame.
# 
# ---
# 
# ### `fe_df`
# 
# The copied DataFrame is stored in a new variable named `fe_df`.
# 
# ```python
# fe_df = clean_df.copy()
# ```
# 
# Now you have:
# 
# - `clean_df` → Original cleaned dataset.
# - `fe_df` → Dataset used for feature engineering.
# 
# ---
# 
# ## Why Use `.copy()`?
# 
# Creating a copy helps to:
# 
# - Preserve the original cleaned dataset.
# - Prevent accidental modifications.
# - Allow experimentation with new features.
# - Make debugging easier if something goes wrong.
# 
# This is considered a **best practice** in data analysis and machine learning.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# clean_df = pd.DataFrame({
#     "Item": ["Laptop", "Phone", "Printer"],
#     "Quantity": [2, 5, 3]
# })
# 
# # Create a copy
# fe_df = clean_df.copy()
# 
# # Modify the copied DataFrame
# fe_df["Quantity"] = fe_df["Quantity"] * 2
# 
# print("Original DataFrame")
# print(clean_df)
# 
# print("\nCopied DataFrame")
# print(fe_df)
# ```
# 
# ### Output
# 
# ```text
# Original DataFrame
# 
#       Item  Quantity
# 0   Laptop         2
# 1    Phone         5
# 2  Printer         3
# 
# Copied DataFrame
# 
#       Item  Quantity
# 0   Laptop         4
# 1    Phone        10
# 2  Printer         6
# ```
# 
# Notice that changing `fe_df` **does not** change `clean_df`.
# 
# ---
# 
# ## `.copy()` vs Assignment (`=`)
# 
# ### Without `.copy()`
# 
# ```python
# fe_df = clean_df
# ```
# 
# Both variables refer to the **same DataFrame**.
# 
# If you modify `fe_df`:
# 
# ```python
# fe_df["Quantity"] = 0
# ```
# 
# Then `clean_df` is also modified.
# 
# ---
# 
# ### With `.copy()`
# 
# ```python
# fe_df = clean_df.copy()
# ```
# 
# A completely separate DataFrame is created.
# 
# Changes made to `fe_df` do **not** affect `clean_df`.
# 
# ---
# 
# ## Key Takeaways
# 
# - `.copy()` creates an independent copy of a DataFrame.
# - `fe_df` is commonly used to represent a **Feature Engineering DataFrame**.
# - Using `.copy()` protects the original cleaned dataset from accidental changes.
# - Creating separate DataFrames for cleaning, feature engineering, and modeling is a best practice in data science and machine learning workflows.

# In[25]:


# --- Date features ---
fe_df["Year"] = fe_df["Transaction Date"].dt.year
fe_df["Month"] = fe_df["Transaction Date"].dt.month
fe_df["Day"] = fe_df["Transaction Date"].dt.day
fe_df["Quarter"] = fe_df["Transaction Date"].dt.quarter
fe_df["Day of Week"] = fe_df["Transaction Date"].dt.day_name()


# # Feature Engineering: Extracting Date Features
# 
# ## Code
# 
# ```python
# # --- Date features ---
# fe_df["Year"] = fe_df["Transaction Date"].dt.year
# fe_df["Month"] = fe_df["Transaction Date"].dt.month
# fe_df["Day"] = fe_df["Transaction Date"].dt.day
# fe_df["Quarter"] = fe_df["Transaction Date"].dt.quarter
# fe_df["Day of Week"] = fe_df["Transaction Date"].dt.day_name()
# ```
# 
# ## Explanation
# 
# This code performs **feature engineering** by extracting useful information from the **Transaction Date** column.
# 
# Instead of using only the complete date, the code creates several new columns that can be used for analysis, reporting, and machine learning.
# 
# The newly created features are:
# 
# - **Year**
# - **Month**
# - **Day**
# - **Quarter**
# - **Day of Week**
# 
# These features help identify trends and patterns in sales over time.
# 
# ---
# 
# ## What is Feature Engineering?
# 
# **Feature engineering** is the process of creating new variables (features) from existing data to improve analysis and machine learning models.
# 
# In this example, several new features are created from the **Transaction Date** column.
# 
# ---
# 
# ## The `.dt` Accessor
# 
# The `.dt` accessor is used to access properties of a **datetime** column.
# 
# ```python
# fe_df["Transaction Date"].dt
# ```
# 
# It allows you to extract different parts of a date, such as:
# 
# - Year
# - Month
# - Day
# - Quarter
# - Weekday
# - Hour
# - Minute
# 
# ---
# 
# ## 1. Extract the Year
# 
# ```python
# fe_df["Year"] = fe_df["Transaction Date"].dt.year
# ```
# 
# This creates a new column called **Year** containing the year of each transaction.
# 
# ### Example
# 
# | Transaction Date | Year |
# |-----------------|-----:|
# | 2026-01-15 | 2026 |
# | 2025-08-10 | 2025 |
# | 2024-12-05 | 2024 |
# 
# ### Why is it useful?
# 
# - Analyze yearly sales.
# - Compare annual performance.
# - Identify long-term business trends.
# 
# ---
# 
# ## 2. Extract the Month
# 
# ```python
# fe_df["Month"] = fe_df["Transaction Date"].dt.month
# ```
# 
# This extracts the month as a number between **1 and 12**.
# 
# ### Example
# 
# | Transaction Date | Month |
# |-----------------|------:|
# | 2026-01-15 | 1 |
# | 2026-04-20 | 4 |
# | 2026-11-05 | 11 |
# 
# ### Why is it useful?
# 
# - Analyze monthly sales.
# - Identify seasonal trends.
# - Compare performance across months.
# 
# ---
# 
# ## 3. Extract the Day
# 
# ```python
# fe_df["Day"] = fe_df["Transaction Date"].dt.day
# ```
# 
# This extracts the day of the month.
# 
# ### Example
# 
# | Transaction Date | Day |
# |-----------------|----:|
# | 2026-01-15 | 15 |
# | 2026-04-20 | 20 |
# | 2026-11-05 | 5 |
# 
# ### Why is it useful?
# 
# - Analyze daily transactions.
# - Identify peak shopping days.
# - Detect unusual daily patterns.
# 
# ---
# 
# ## 4. Extract the Quarter
# 
# ```python
# fe_df["Quarter"] = fe_df["Transaction Date"].dt.quarter
# ```
# 
# This extracts the quarter of the year.
# 
# The four quarters are:
# 
# | Quarter | Months |
# |---------|--------|
# | Q1 | January – March |
# | Q2 | April – June |
# | Q3 | July – September |
# | Q4 | October – December |
# 
# ### Example
# 
# | Transaction Date | Quarter |
# |-----------------|---------:|
# | 2026-02-10 | 1 |
# | 2026-05-25 | 2 |
# | 2026-08-18 | 3 |
# | 2026-11-12 | 4 |
# 
# ### Why is it useful?
# 
# - Prepare quarterly business reports.
# - Compare sales across quarters.
# - Track financial performance.
# 
# ---
# 
# ## 5. Extract the Day of the Week
# 
# ```python
# fe_df["Day of Week"] = fe_df["Transaction Date"].dt.day_name()
# ```
# 
# This extracts the full name of the weekday.
# 
# ### Example
# 
# | Transaction Date | Day of Week |
# |-----------------|-------------|
# | 2026-01-05 | Monday |
# | 2026-01-06 | Tuesday |
# | 2026-01-10 | Saturday |
# 
# ### Why is it useful?
# 
# - Identify the busiest shopping days.
# - Compare weekday and weekend sales.
# - Optimize staffing and inventory planning.
# 
# ---
# 
# ## Complete Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Transaction Date": pd.to_datetime([
#         "2026-01-15",
#         "2026-05-20",
#         "2026-10-05"
#     ])
# })
# 
# df["Year"] = df["Transaction Date"].dt.year
# df["Month"] = df["Transaction Date"].dt.month
# df["Day"] = df["Transaction Date"].dt.day
# df["Quarter"] = df["Transaction Date"].dt.quarter
# df["Day of Week"] = df["Transaction Date"].dt.day_name()
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#   Transaction Date  Year  Month  Day  Quarter Day of Week
# 0       2026-01-15  2026      1   15        1     Thursday
# 1       2026-05-20  2026      5   20        2    Wednesday
# 2       2026-10-05  2026     10    5        4       Monday
# ```
# 
# ---
# 
# ## Benefits of Extracting Date Features
# 
# Creating separate date features allows you to:
# 
# - Analyze yearly, quarterly, monthly, and daily trends.
# - Identify seasonal sales patterns.
# - Compare weekday versus weekend performance.
# - Build more accurate machine learning models.
# - Create interactive dashboards with time-based filters.
# 
# ---
# 
# ## Key Takeaways
# 
# - The `.dt` accessor extracts components from a datetime column.
# - `dt.year` extracts the year.
# - `dt.month` extracts the month (1–12).
# - `dt.day` extracts the day of the month.
# - `dt.quarter` extracts the quarter (1–4).
# - `dt.day_name()` returns the full weekday name.
# - Extracting date features is a common feature engineering technique that improves data analysis, visualization, and predictive modeling.

# In[26]:


# --- Spending Category (based on Total Spent) ---
def spending_category(x):
    if x < 100:
        return "Low"
    elif x <= 300:
        return "Medium"
    else:
        return "High"

fe_df["Spending Category"] = fe_df["Total Spent"].apply(spending_category)


# # Feature Engineering: Creating a Spending Category
# 
# ## Code
# 
# ```python
# # --- Spending Category (based on Total Spent) ---
# def spending_category(x):
#     if x < 100:
#         return "Low"
#     elif x <= 300:
#         return "Medium"
#     else:
#         return "High"
# 
# fe_df["Spending Category"] = fe_df["Total Spent"].apply(spending_category)
# ```
# 
# ## Explanation
# 
# This code creates a new feature called **Spending Category** by classifying each transaction based on its **Total Spent** value.
# 
# Instead of working with continuous numerical values, transactions are grouped into meaningful categories:
# 
# - **Low**
# - **Medium**
# - **High**
# 
# This process is known as **categorization** or **binning**, and it makes the data easier to analyze and visualize.
# 
# ---
# 
# ## What is Feature Engineering?
# 
# Feature engineering is the process of creating new variables (features) from existing data to improve analysis, reporting, and machine learning models.
# 
# In this example, the **Total Spent** column is transformed into a categorical feature called **Spending Category**.
# 
# ---
# 
# ## Step 1: Define a Function
# 
# ```python
# def spending_category(x):
# ```
# 
# The `def` keyword is used to create a **function**.
# 
# A function is a reusable block of code that performs a specific task.
# 
# Here:
# 
# - **Function name:** `spending_category`
# - **Parameter:** `x`
# 
# The variable `x` represents the **Total Spent** for one transaction.
# 
# ---
# 
# ## Step 2: Apply Conditional Statements
# 
# ### Low Spending
# 
# ```python
# if x < 100:
#     return "Low"
# ```
# 
# If the total amount spent is **less than 100**, the function returns:
# 
# ```text
# Low
# ```
# 
# Example:
# 
# | Total Spent | Category |
# |-------------|----------|
# | 50 | Low |
# | 80 | Low |
# 
# ---
# 
# ### Medium Spending
# 
# ```python
# elif x <= 300:
#     return "Medium"
# ```
# 
# The `elif` statement means **"else if"**.
# 
# If the amount is **between 100 and 300 (inclusive)**, the function returns:
# 
# ```text
# Medium
# ```
# 
# Example:
# 
# | Total Spent | Category |
# |-------------|----------|
# | 120 | Medium |
# | 250 | Medium |
# | 300 | Medium |
# 
# ---
# 
# ### High Spending
# 
# ```python
# else:
#     return "High"
# ```
# 
# If the amount is **greater than 300**, the function returns:
# 
# ```text
# High
# ```
# 
# Example:
# 
# | Total Spent | Category |
# |-------------|----------|
# | 350 | High |
# | 600 | High |
# 
# ---
# 
# ## Summary of the Rules
# 
# | Total Spent | Spending Category |
# |-------------|-------------------|
# | Less than 100 | Low |
# | 100 to 300 | Medium |
# | Greater than 300 | High |
# 
# ---
# 
# ## Step 3: Apply the Function to the Column
# 
# ```python
# fe_df["Spending Category"] = fe_df["Total Spent"].apply(spending_category)
# ```
# 
# ### `.apply()`
# 
# The `apply()` function applies another function to every value in a column.
# 
# ```python
# .apply(spending_category)
# ```
# 
# For each value in **Total Spent**:
# 
# 1. The value is passed to `spending_category()`.
# 2. The function determines whether it is **Low**, **Medium**, or **High**.
# 3. The result is stored in the new **Spending Category** column.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Total Spent": [50, 150, 320, 95, 275]
# })
# 
# def spending_category(x):
#     if x < 100:
#         return "Low"
#     elif x <= 300:
#         return "Medium"
#     else:
#         return "High"
# 
# df["Spending Category"] = df["Total Spent"].apply(spending_category)
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Total Spent Spending Category
# 0           50               Low
# 1          150            Medium
# 2          320              High
# 3           95               Low
# 4          275            Medium
# ```
# 
# ---
# 
# ## Why Create Spending Categories?
# 
# Creating spending categories helps to:
# 
# - Segment customers based on spending behavior.
# - Identify high-value customers.
# - Compare purchasing patterns across different groups.
# - Create clearer charts and dashboards.
# - Improve machine learning models by using categorical features.
# 
# ---
# 
# ## Key Takeaways
# 
# - `def` creates a reusable function.
# - `if`, `elif`, and `else` define the conditions for assigning categories.
# - `.apply()` applies the function to every value in a DataFrame column.
# - The new **Spending Category** feature groups transactions into **Low**, **Medium**, and **High** spending levels.
# - Categorizing numerical data into meaningful groups is a common feature engineering technique used in data analysis and machine learning.

# In[27]:


# --- Quantity Category ---
def quantity_category(q):
    if q <= 3:
        return "Small Purchase"
    elif q <= 7:
        return "Medium Purchase"
    else:
        return "Bulk Purchase"

fe_df["Quantity Category"] = fe_df["Quantity"].apply(quantity_category)


# # Feature Engineering: Creating a Quantity Category
# 
# ## Code
# 
# ```python
# # --- Quantity Category ---
# def quantity_category(q):
#     if q <= 3:
#         return "Small Purchase"
#     elif q <= 7:
#         return "Medium Purchase"
#     else:
#         return "Bulk Purchase"
# 
# fe_df["Quantity Category"] = fe_df["Quantity"].apply(quantity_category)
# ```
# 
# ## Explanation
# 
# This code creates a new feature called **Quantity Category** by grouping transactions according to the number of items purchased (**Quantity**).
# 
# Instead of analyzing raw numerical quantities, purchases are classified into three meaningful categories:
# 
# - **Small Purchase**
# - **Medium Purchase**
# - **Bulk Purchase**
# 
# This process is called **categorization** or **binning**, and it makes the data easier to understand, analyze, and visualize.
# 
# ---
# 
# ## What is Feature Engineering?
# 
# Feature engineering is the process of creating new variables (features) from existing data to improve data analysis, reporting, and machine learning models.
# 
# In this example, the **Quantity** column is transformed into a categorical feature called **Quantity Category**.
# 
# ---
# 
# ## Step 1: Define a Function
# 
# ```python
# def quantity_category(q):
# ```
# 
# The `def` keyword defines a **function**.
# 
# A function is a reusable block of code that performs a specific task.
# 
# Here:
# 
# - **Function name:** `quantity_category`
# - **Parameter:** `q`
# 
# The variable `q` represents the **Quantity** for a single transaction.
# 
# ---
# 
# ## Step 2: Apply Conditional Statements
# 
# ### Small Purchase
# 
# ```python
# if q <= 3:
#     return "Small Purchase"
# ```
# 
# If the quantity purchased is **3 or fewer items**, the function returns:
# 
# ```text
# Small Purchase
# ```
# 
# Example:
# 
# | Quantity | Category |
# |---------:|----------|
# | 1 | Small Purchase |
# | 2 | Small Purchase |
# | 3 | Small Purchase |
# 
# ---
# 
# ### Medium Purchase
# 
# ```python
# elif q <= 7:
#     return "Medium Purchase"
# ```
# 
# The `elif` statement means **"else if"**.
# 
# If the quantity is **between 4 and 7 items (inclusive)**, the function returns:
# 
# ```text
# Medium Purchase
# ```
# 
# Example:
# 
# | Quantity | Category |
# |---------:|----------|
# | 4 | Medium Purchase |
# | 6 | Medium Purchase |
# | 7 | Medium Purchase |
# 
# ---
# 
# ### Bulk Purchase
# 
# ```python
# else:
#     return "Bulk Purchase"
# ```
# 
# If the quantity is **greater than 7**, the function returns:
# 
# ```text
# Bulk Purchase
# ```
# 
# Example:
# 
# | Quantity | Category |
# |---------:|----------|
# | 8 | Bulk Purchase |
# | 12 | Bulk Purchase |
# | 20 | Bulk Purchase |
# 
# ---
# 
# ## Summary of the Rules
# 
# | Quantity | Quantity Category |
# |----------|-------------------|
# | 1–3 | Small Purchase |
# | 4–7 | Medium Purchase |
# | More than 7 | Bulk Purchase |
# 
# ---
# 
# ## Step 3: Apply the Function to the Column
# 
# ```python
# fe_df["Quantity Category"] = fe_df["Quantity"].apply(quantity_category)
# ```
# 
# ### `.apply()`
# 
# The `apply()` function applies another function to every value in a DataFrame column.
# 
# ```python
# .apply(quantity_category)
# ```
# 
# For each value in the **Quantity** column:
# 
# 1. The quantity is passed to `quantity_category()`.
# 2. The function determines the appropriate category.
# 3. The resulting category is stored in the new **Quantity Category** column.
# 
# ---
# 
# ## Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Quantity": [2, 5, 9, 3, 7, 12]
# })
# 
# def quantity_category(q):
#     if q <= 3:
#         return "Small Purchase"
#     elif q <= 7:
#         return "Medium Purchase"
#     else:
#         return "Bulk Purchase"
# 
# df["Quantity Category"] = df["Quantity"].apply(quantity_category)
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Quantity Quantity Category
# 0         2    Small Purchase
# 1         5   Medium Purchase
# 2         9     Bulk Purchase
# 3         3    Small Purchase
# 4         7   Medium Purchase
# 5        12     Bulk Purchase
# ```
# 
# ---
# 
# ## Why Create Quantity Categories?
# 
# Grouping quantities into categories helps to:
# 
# - Understand customer buying behavior.
# - Identify bulk buyers.
# - Compare purchasing patterns across customer groups.
# - Create clearer charts and dashboards.
# - Improve machine learning models by using meaningful categorical features.
# 
# ---
# 
# ## Key Takeaways
# 
# - `def` creates a reusable function.
# - `if`, `elif`, and `else` define the conditions for categorizing purchases.
# - `.apply()` applies the function to every value in the **Quantity** column.
# - The new **Quantity Category** feature groups purchases into **Small**, **Medium**, and **Bulk** categories.
# - Categorizing numerical data into meaningful groups is a common feature engineering technique used in data analysis and machine learning.

# In[29]:


fe_df.head()


# In[28]:


# --- Discount Status ---
fe_df["Discount Status"] = fe_df["Discount Applied"].map({True: "Discounted", False: "Not Discounted"})


# # Feature Engineering: Creating a Discount Status Column
# 
# ## Code
# 
# ```python
# # --- Discount Status ---
# fe_df["Discount Status"] = fe_df["Discount Applied"].map(
#     {True: "Discounted", False: "Not Discounted"}
# )
# ```
# 
# ## Explanation
# 
# This code creates a new feature called **Discount Status** by converting the Boolean values in the **Discount Applied** column into more descriptive text labels.
# 
# Instead of displaying **True** and **False**, the new column contains:
# 
# - **Discounted**
# - **Not Discounted**
# 
# This makes the dataset easier to understand, especially when creating reports, dashboards, and charts.
# 
# ---
# 
# ## What is Feature Engineering?
# 
# Feature engineering is the process of creating new variables (features) from existing data to improve analysis, reporting, and machine learning models.
# 
# In this example, the Boolean **Discount Applied** column is transformed into a more readable categorical feature.
# 
# ---
# 
# ## Breaking Down the Code
# 
# ### `fe_df["Discount Applied"]`
# 
# This selects the **Discount Applied** column.
# 
# ```python
# fe_df["Discount Applied"]
# ```
# 
# Example:
# 
# | Discount Applied |
# |------------------|
# | True |
# | False |
# | True |
# | False |
# 
# ---
# 
# ### `.map()`
# 
# The `map()` function replaces existing values with new values according to a specified mapping.
# 
# ```python
# .map({...})
# ```
# 
# It works by matching each value in the column with a corresponding value in a dictionary.
# 
# ---
# 
# ### The Dictionary
# 
# ```python
# {
#     True: "Discounted",
#     False: "Not Discounted"
# }
# ```
# 
# This dictionary defines how each Boolean value should be converted.
# 
# | Original Value | New Value |
# |---------------|-----------|
# | True | Discounted |
# | False | Not Discounted |
# 
# ---
# 
# ### Creating the New Column
# 
# ```python
# fe_df["Discount Status"] = ...
# ```
# 
# The transformed values are stored in a new column called **Discount Status**.
# 
# The original **Discount Applied** column remains unchanged.
# 
# ---
# 
# ## Example
# 
# ### Before Transformation
# 
# | Discount Applied |
# |------------------|
# | True |
# | False |
# | True |
# | False |
# 
# ### After Transformation
# 
# | Discount Applied | Discount Status |
# |------------------|-----------------|
# | True | Discounted |
# | False | Not Discounted |
# | True | Discounted |
# | False | Not Discounted |
# 
# ---
# 
# ## Complete Example
# 
# ```python
# import pandas as pd
# 
# df = pd.DataFrame({
#     "Discount Applied": [True, False, True, False]
# })
# 
# df["Discount Status"] = df["Discount Applied"].map({
#     True: "Discounted",
#     False: "Not Discounted"
# })
# 
# print(df)
# ```
# 
# ### Output
# 
# ```text
#    Discount Applied  Discount Status
# 0              True       Discounted
# 1             False   Not Discounted
# 2              True       Discounted
# 3             False   Not Discounted
# ```
# 
# ---
# 
# ## Why Use `.map()`?
# 
# The `map()` function is useful because it:
# 
# - Converts coded values into meaningful labels.
# - Makes reports and dashboards easier to interpret.
# - Improves the readability of the dataset.
# - Simplifies grouping and visualization based on categories.
# 
# For example, a bar chart labeled **Discounted** and **Not Discounted** is much easier to understand than one labeled **True** and **False**.
# 
# ---
# 
# ## Key Takeaways
# 
# - `.map()` replaces existing values using a dictionary.
# - Boolean values (`True` and `False`) are converted into descriptive text labels.
# - A new feature called **Discount Status** is created while preserving the original **Discount Applied** column.
# - Converting coded values into readable categories improves data presentation, reporting, and visualization.

# In[30]:


# --- Revenue Band (tertiles of Total Spent -> Bronze/Silver/Gold) ---
fe_df["Revenue Band"] = pd.qcut(
    fe_df["Total Spent"], q=3, labels=["Bronze", "Silver", "Gold"]
)

fe_df[["Transaction Date", "Year", "Month", "Day", "Quarter", "Day of Week",
       "Total Spent", "Spending Category", "Quantity Category",
       "Discount Status", "Revenue Band"]].head()


# # Feature Engineering: Creating Revenue Bands Using `qcut()`
# 
# ## Code
# 
# ```python
# # --- Revenue Band (tertiles of Total Spent -> Bronze/Silver/Gold) ---
# fe_df["Revenue Band"] = pd.qcut(
#     fe_df["Total Spent"],
#     q=3,
#     labels=["Bronze", "Silver", "Gold"]
# )
# 
# fe_df[[
#     "Transaction Date",
#     "Year",
#     "Month",
#     "Day",
#     "Quarter",
#     "Day of Week",
#     "Total Spent",
#     "Spending Category",
#     "Quantity Category",
#     "Discount Status",
#     "Revenue Band"
# ]].head()
# ```
# 
# ## Explanation
# 
# This code creates a new feature called **Revenue Band** by dividing the **Total Spent** values into **three equal-sized groups** (tertiles).
# 
# Each transaction is assigned one of the following labels:
# 
# - **Bronze** – Lowest spending group
# - **Silver** – Middle spending group
# - **Gold** – Highest spending group
# 
# This helps categorize customers or transactions based on their spending level.
# 
# ---
# 
# # What is Feature Engineering?
# 
# Feature engineering is the process of creating new variables (features) from existing data to improve data analysis, reporting, and machine learning models.
# 
# In this example, the numerical **Total Spent** column is converted into a categorical variable called **Revenue Band**.
# 
# ---
# 
# # Breaking Down the Code
# 
# ## `pd.qcut()`
# 
# ```python
# pd.qcut()
# ```
# 
# `qcut()` stands for **Quantile Cut**.
# 
# It divides a numerical column into groups containing **approximately the same number of observations**.
# 
# Unlike `cut()`, which creates groups based on fixed value ranges, `qcut()` creates groups based on the **distribution of the data**.
# 
# ---
# 
# ## `fe_df["Total Spent"]`
# 
# ```python
# fe_df["Total Spent"]
# ```
# 
# This is the numerical column that will be divided into revenue groups.
# 
# Example:
# 
# | Total Spent |
# |-------------|
# | 45 |
# | 80 |
# | 120 |
# | 180 |
# | 250 |
# | 400 |
# | 520 |
# | 750 |
# | 980 |
# 
# ---
# 
# ## `q=3`
# 
# ```python
# q=3
# ```
# 
# The parameter `q` specifies the number of groups (quantiles) to create.
# 
# Since:
# 
# ```python
# q = 3
# ```
# 
# The data is divided into **three equal-sized groups**, called **tertiles**.
# 
# Each group contains approximately one-third of the observations.
# 
# ---
# 
# ## `labels=["Bronze", "Silver", "Gold"]`
# 
# ```python
# labels=["Bronze", "Silver", "Gold"]
# ```
# 
# Instead of assigning numbers (0, 1, 2), the groups receive meaningful labels.
# 
# | Revenue Band | Description |
# |---------------|------------|
# | Bronze | Lowest one-third of spending values |
# | Silver | Middle one-third of spending values |
# | Gold | Highest one-third of spending values |
# 
# ---
# 
# ## Example
# 
# Suppose the **Total Spent** values are:
# 
# | Total Spent |
# |-------------|
# | 40 |
# | 70 |
# | 95 |
# | 120 |
# | 180 |
# | 250 |
# | 320 |
# | 500 |
# | 900 |
# 
# Using:
# 
# ```python
# pd.qcut(df["Total Spent"], q=3,
#         labels=["Bronze", "Silver", "Gold"])
# ```
# 
# The result could be:
# 
# | Total Spent | Revenue Band |
# |-------------|--------------|
# | 40 | Bronze |
# | 70 | Bronze |
# | 95 | Bronze |
# | 120 | Silver |
# | 180 | Silver |
# | 250 | Silver |
# | 320 | Gold |
# | 500 | Gold |
# | 900 | Gold |
# 
# Notice that each band contains approximately the same **number of records**, not the same spending range.
# 
# ---
# 
# # Display Selected Columns
# 
# ```python
# fe_df[[ ... ]].head()
# ```
# 
# This selects only the specified columns and displays the **first five rows** of the DataFrame.
# 
# The selected columns are:
# 
# - Transaction Date
# - Year
# - Month
# - Day
# - Quarter
# - Day of Week
# - Total Spent
# - Spending Category
# - Quantity Category
# - Discount Status
# - Revenue Band
# 
# ---
# 
# ## `.head()`
# 
# ```python
# .head()
# ```
# 
# The `head()` function displays the first **five rows** of a DataFrame.
# 
# Example output:
# 
# | Transaction Date | Year | Month | Day | Quarter | Day of Week | Total Spent | Spending Category | Quantity Category | Discount Status | Revenue Band |
# |------------------|-----:|------:|----:|--------:|-------------|------------:|-------------------|-------------------|-----------------|--------------|
# | 2026-01-05 | 2026 | 1 | 5 | 1 | Monday | 85 | Low | Small Purchase | Discounted | Bronze |
# | 2026-02-10 | 2026 | 2 | 10 | 1 | Tuesday | 180 | Medium | Medium Purchase | Not Discounted | Silver |
# | 2026-03-15 | 2026 | 3 | 15 | 1 | Sunday | 520 | High | Bulk Purchase | Discounted | Gold |
# | 2026-04-18 | 2026 | 4 | 18 | 2 | Saturday | 95 | Low | Small Purchase | Not Discounted | Bronze |
# | 2026-05-22 | 2026 | 5 | 22 | 2 | Friday | 260 | Medium | Medium Purchase | Discounted | Silver |
# 
# ---
# 
# # Why Use Revenue Bands?
# 
# Creating revenue bands helps to:
# 
# - Segment customers by spending level.
# - Identify high-value customers.
# - Compare sales across different revenue groups.
# - Create more informative dashboards and visualizations.
# - Improve machine learning models by using categorical features.
# 
# ---
# 
# # `qcut()` vs `cut()`
# 
# | `pd.cut()` | `pd.qcut()` |
# |-------------|-------------|
# | Divides data into fixed value ranges. | Divides data into groups with approximately equal numbers of observations. |
# | Group sizes may differ. | Group sizes are approximately equal. |
# | Suitable when value ranges are predefined. | Suitable for customer segmentation and ranking. |
# 
# Example:
# 
# ```python
# pd.cut(df["Total Spent"], bins=[0,100,300,1000])
# ```
# 
# Creates fixed spending ranges.
# 
# ```python
# pd.qcut(df["Total Spent"], q=3)
# ```
# 
# Creates three groups with roughly equal numbers of records.
# 
# ---
# 
# # Key Takeaways
# 
# - `pd.qcut()` divides numerical data into **quantiles** with approximately equal numbers of observations.
# - `q=3` creates **three tertiles**.
# - The labels **Bronze**, **Silver**, and **Gold** replace numeric group identifiers, making the data easier to interpret.
# - `.head()` displays the first five rows of the selected columns, allowing you to quickly verify the newly created features.
# - Revenue bands are useful for customer segmentation, sales analysis, reporting, dashboards, and machine learning.

# In[31]:


# Answer the Part 3 questions
year_rev = fe_df.groupby("Year")["Total Spent"].sum().sort_values(ascending=False)
month_rev = fe_df.groupby("Month")["Total Spent"].sum().sort_values(ascending=False)
cat_rev = fe_df.groupby("Category")["Total Spent"].sum().sort_values(ascending=False)
item_rev = fe_df.groupby("Item")["Total Spent"].sum().sort_values(ascending=False)
loc_rev = fe_df.groupby("Location")["Total Spent"].sum().sort_values(ascending=False)

print("Revenue by Year:\n", year_rev, "\n")
print("Revenue by Month:\n", month_rev, "\n")
print("Revenue by Category:\n", cat_rev, "\n")
print("Top 5 Items by Revenue:\n", item_rev.head(), "\n")
print("Revenue by Location:\n", loc_rev)


# In[ ]:




