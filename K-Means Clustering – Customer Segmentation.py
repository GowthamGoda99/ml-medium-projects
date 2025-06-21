import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Working Gist URL
url = "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv"
df = pd.read_csv(url)

# Data overview
print(df.head())

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.show()

# Final clustering with k=5
kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
df["cluster"] = kmeans.fit_predict(X)

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="cluster",
    palette="tab10"
)
plt.title("Customer Segments")
plt.show()
