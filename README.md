**🛍️ Mall Customer Segmentation using K-Means Clustering**

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to segment customers based on their **Annual Income** and **Spending Score** using the **Mall Customers dataset**.
It helps in identifying customer groups for targeted marketing strategies and better business decision-making.

**📁 Dataset Overview**
- **Filename:** Mall_Customers.csv
- **Source:** Kaggle
- **Columns Used for Clustering:**
  - Annual Income (k$)
  - Spending Score (1-100)

**📌 Problem Statement**

The goal is to group customers into **distinct segments** based on their **purchasing behavior** using **K-Means clustering**, and visualize the resulting clusters to understand different customer types (e.g., high-income low-spenders, low-income high-spenders, etc.).

**🚀 How It Works**

**1. Importing Libraries**

Used libraries include:
- pandas, numpy for data handling
- matplotlib, seaborn for visualization
- scikit-learn for clustering and preprocessing

**2. Loading the Dataset**
```
df = pd.read_csv('/content/sample_data/Mall_Customers.csv')
```
Basic dataset inspection using:
```
df.head()
df.info()
df.columns
```

**3. Feature Selection**
- Selected two key numerical features for clustering:
```
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

**4. Feature Scaling**
- Scaled the features for better clustering performance using **StandardScaler**:
```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**5. Finding Optimal Clusters – Elbow Method**
- The **elbow method** is used to determine the optimal number of clusters **(k)**:
```
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X_scaled)
  wcss.append(kmeans.inertia_)
```
Visualized using:
```
plt.plot(range(1,11), wcss, marker='o')
```
- The **"elbow"** point (where the WCSS(Within-Cluster Sum of Squares) drop starts to level off) indicates the best k.

**6. Applying K-Means Clustering**
- After determining **k=5**, we fit the model:
```
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
```
- Cluster labels are then added to the original DataFrame.

**📊 Visualization of Clusters**
- Each cluster is plotted in a different color.
- Centroids are shown using yellow ‘X’ markers.

**Resulting plot:** Customer Segments based on **Annual Income vs. Spending Score**.

**📎 Requirements**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them using:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

**📈 Sample Output (Cluster Plot)**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):

 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Gender                  200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 

dtypes: int64(4), object(1)
memory usage: 7.9+ KB
```
![Screenshot (109)](https://github.com/user-attachments/assets/9412405b-2bdc-4eca-9850-961dcf7ee149)

![Screenshot (110)](https://github.com/user-attachments/assets/91fc532b-1a0d-494a-bca9-0b09ae901b81)

**📚 Learnings**
- Hands-on implementation of K-Means clustering
- Understanding and using the elbow method
- Visualizing high-dimensional data clusters
- Importance of feature scaling

**🧠 Future Improvements**
- Include more features like Age, Gender
- Try other clustering algorithms (DBSCAN, Hierarchical)
- Build an interactive dashboard using Plotly or Streamlit

**👨‍💻 Author**

**Yash Patel**

Python Enthusiast | Machine Learning Explorer | Aspiring Data Scientist
