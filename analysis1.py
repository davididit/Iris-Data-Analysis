#Data Analysis

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Iris.csv")

# Display the first five rows
print("First five rows:")
print(data.head())

# Display the number of rows and columns
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns)

# Display dataset information
print("\nDataset Information:")
data.info()

# Calculate summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Count each species
print("\nSpecies Count:")
print(data["Species"].value_counts())

# Plot 1: Histogram
data.select_dtypes(include="number").hist(figsize=(10, 8))
plt.suptitle("Iris Dataset Histograms")
plt.tight_layout()
plt.show()

# Plot 2: Scatter Plot
sns.scatterplot(
    data=data,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species"
)
plt.title("Sepal Length vs Petal Length")
plt.show()

# Plot 3: Box Plot
sns.boxplot(
    data=data,
    x="Species",
    y="PetalLengthCm"
)
plt.title("Petal Length by Species")
plt.show()

# Plot 4: Correlation Heatmap
numeric_data = data.select_dtypes(include="number")

sns.heatmap(
    numeric_data.drop(columns=["Id"], errors="ignore").corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()