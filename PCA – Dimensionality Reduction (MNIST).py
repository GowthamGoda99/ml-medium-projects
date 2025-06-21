# MEDIUM – 04. PCA – Dimensionality Reduction (MNIST)
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df["digit"] = y

sns.scatterplot(data=df, x="PC1", y="PC2", hue="digit", palette="tab10", alpha=0.7)
plt.title("PCA - MNIST Digits (2D)")
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_.sum())
