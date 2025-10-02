# News Clustering with KMeans & Hierarchical Agglomerative Clustering  

##  Description  
This project implements an **Unsupervised Machine Learning pipeline** to cluster news articles into groups based on their content.  
We apply **TF-IDF vectorization** for text representation, **PCA** for dimensionality reduction, and then perform clustering using:  
- **KMeans Clustering**  
- **Agglomerative Hierarchical Clustering**  

After evaluating the models using the **Silhouette Score**, we concluded that the **best model is Agglomerative Hierarchical Clustering**, as it has the **highest silhouette score**.  

Finally, we tested the models with **sample news articles** to predict their cluster assignments.  

---

##  Dataset  
The dataset (`news_data.tsv`) contains news articles with different **labels** (truth categories):  
- half-true  
- false  
- mostly-true  
- true  
- barely-true  
- pants-fire  

### Steps:  
1. Removed unnecessary columns.  
2. Balanced the dataset by sampling **34 entries per class** → total **204 datapoints**.  
3. Preprocessed text: lowercasing, tokenization, stopword removal, stemming.  
4. Converted text to numerical vectors using **TF-IDF**.  
5. Applied **PCA (2 components)** to reduce dimensionality for visualization and clustering.  

---

## Methodology  

### 🔹 Preprocessing  
- Tokenization (`nltk.word_tokenize`)  
- Removing stopwords & punctuation  
- Stemming (`PorterStemmer`)  
- Vectorization with **TF-IDF**  

### 🔹 Clustering  
1. **KMeans Algorithm**  
   - Used **Elbow Method** with Within-Cluster Sum of Squares (WCSS)  
   - Optimal clusters ≈ **6**  
   - Evaluated using **Silhouette Score**  

2. **Agglomerative Hierarchical Clustering**  
   - Built **Dendrogram**  
   - Optimal clusters ≈ **3**  
   - Evaluated with **Silhouette Score**  

---

## Results  

| Algorithm                        | Silhouette Score |
|----------------------------------|------------------|
| KMeans Clustering                | `0.358131`       |
| Agglomerative Hierarchical       | `0.523700`       |

✅ **Agglomerative Hierarchical Clustering has the highest silhouette score**, indicating that it is the **best clustering algorithm** for this dataset.  

---

## Visualizations  

- **Elbow Method for KMeans**  
- **KMeans Clusters (scatter plot with centroids)**  
- **Hierarchical Dendrogram**  
- **Agglomerative Clustering (scatter plot)**  

---

## Testing with Sample Data  

We tested clustering on unseen news text samples:  

```text
"The Chicago Bears have had more starting quarterbacks in the last 10 years than the total number of tenured (UW) faculty fired during the last two decades."  
→ Cluster 2  

"Health care reform legislation is likely to mandate free sex change surgeries."  
→ Cluster 0  

"Jim Dunnam has not lived in the district he represents for years now."  
→ Cluster 1
```
✅ Each input news gets assigned to a cluster number.

---

## Setup & Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/GhadeerZahwe/News-Clustering-Using-KMeans.git
cd News-Clustering-Using-KMeans
```
2. **Create & activate a virtual environment**  
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
```
3. **Install dependencies**  
```bash
pip install -r requirements.txt
```
---
## Tech Stack

- **Python**
- **NLTK** (preprocessing)
- **Scikit-learn** (TF-IDF, PCA, KMeans, Agglomerative Clustering, Silhouette Score)
- **Matplotlib & Seaborn** (visualizations)

---

## Key Learnings

- How to preprocess raw text into numerical vectors using **TF-IDF**
- Dimensionality reduction with **PCA**
- Applying **unsupervised clustering algorithms** to text data
- Using **Silhouette Score** for model evaluation
- Visualizing clustering results

---

## Conclusion

From the evaluation table and visualizations, it is clear that the **silhouette score of Agglomerative Hierarchical Clustering is the highest**, indicating that it is the **best clustering algorithm** for this dataset.

**Therefore, for this project, the optimal number of clusters is 3.**

We also tested the model with **sample news articles**, which were correctly assigned to their respective clusters based on content similarity.

This project demonstrates the power of **unsupervised machine learning** in extracting patterns from **text data** without predefined labels.


