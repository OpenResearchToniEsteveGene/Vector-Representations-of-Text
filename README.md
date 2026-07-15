# Text Vector Representations

A theoretical and empirical comparison of **TF-IDF**, **Latent Semantic Analysis (LSA)**, **Latent Dirichlet Allocation (LDA)**, and **Skip-gram with Negative Sampling (SG-NS)** for text representation and document classification.

The project combines the mathematical development of each technique with custom implementations, optimized high-level library versions, and reproducible experiments on two corpora with very different characteristics: **Sentiment140** and **arXiv**. The **text8** corpus is also used to train general-domain SG-NS word embeddings.

---

## Description

Machine learning algorithms require numerical inputs. Therefore, before documents can be classified, clustered, or retrieved, text must be transformed into vectors that preserve the relevant information contained in the corpus.

This project analyzes four families of text representations:

- **TF-IDF:** a statistical, sparse, and directly interpretable representation.
- **LSA:** an algebraic representation obtained through dimensionality reduction with singular value decomposition.
- **LDA:** a probabilistic representation of documents as mixtures of latent topics.
- **SG-NS:** neural word embeddings aggregated to build dense document vectors.

The comparison is not limited to predictive performance. It also considers:

- dimensionality;
- sparsity;
- memory usage;
- representation construction time;
- classifier training time;
- vector-space geometry;
- interpretability;
- sensitivity to the training domain.

---

## Objectives

The general objective is to analyze, formalize, and compare different text vectorization techniques from both a **theoretical** and an **experimental** perspective.

The specific objectives are:

1. To mathematically formalize TF-IDF, LSA, LDA, and SG-NS.
2. To manually implement the main operations of each method.
3. To compare the custom implementations with optimized `scikit-learn` implementations.
4. To evaluate all representations using a common classifier.
5. To analyze how corpus characteristics affect performance.
6. To study the trade-off between predictive performance, computational cost, dimensionality, and interpretability.
7. To compare general-domain embeddings with domain-specific embeddings.

---

## Methods Analyzed

| Method | Approach | Resulting representation | Main advantage | Main limitation |
|---|---|---|---|---|
| TF-IDF | Statistical | Sparse vector whose dimension is determined by the vocabulary | Simplicity and direct interpretation | High dimensionality and sparsity |
| LSA | Algebraic | Dense vector in a latent semantic space | Dimensionality reduction and co-occurrence modeling | Latent components are less directly interpretable |
| LDA | Probabilistic | Topic distribution for each document | Compact and interpretable representation | High computational cost and dependence on clear topic structure |
| SG-NS | Neural | Dense word and document embeddings | Captures distributional relationships | Depends on the training corpus and loses word order when embeddings are aggregated |

---

## Mathematical Foundations

### TF-IDF

For a term $t$, a document $d$, and a corpus $D$:

$$
tfidf(t,d)=tf(t,d)\cdot idf(t,D)
$$

where

$$
tf(t,d)=\frac{f(t,d)}{\max_{t' \in d} f(t',d)}
$$

and

$$
idf(t,D)=\log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right).
$$

### LSA

LSA starts from a term-document matrix $A$ and applies singular value decomposition:

$$
A = U\Sigma V^\top.
$$

The reduced representation keeps the $k$ components associated with the largest singular values:

$$
A_k = U_k\Sigma_kV_k^\top.
$$

This approximation reduces noise and projects documents into a lower-dimensional semantic space.

### LDA

LDA represents each document through a probability distribution over $K$ topics:

$$
\theta_d =
(\theta_{d,1},\ldots,\theta_{d,K}),
$$

where $\theta_{d,k}$ is the weight of topic $k$ in document $d$.

The custom implementation uses **collapsed Gibbs sampling** to estimate topic assignments and the document-topic and topic-word distributions.

### SG-NS

Skip-gram learns word embeddings by predicting context terms from a central word. Negative sampling replaces the full `softmax` calculation with a binary classification task between positive and negative word-context pairs.

A document vector is built using a TF-IDF-weighted average of word embeddings:

$$
v_D =\frac{\sum_{w \in D}tfidf(w,D)\,v_w}{\sum_{w \in D}tfidf(w,D)}.
$$

---

## Datasets

The datasets are not included directly in the repository because of their size and their respective distribution conditions. They can be obtained from the sources below.

### Sentiment140

Sentiment140 is a corpus of short Twitter messages designed for sentiment analysis.

- Original size: **1,600,000 tweets**.
- Classes: positive and negative sentiment.
- Balanced experimental sample: **20,000 documents**.
- Training split: **16,000 documents**.
- Test split: **4,000 documents**.
- Characteristics: short, informal, noisy, and context-limited texts.

#### Download

- Original project page:  
  <http://help.sentiment140.com/for-students/>

- Original dataset archive:  
  <http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip>

After extracting the archive, the main training file used in this project is:

```text
training.1600000.processed.noemoticon.csv
```

Example:

```bash
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip
```

The original labels are usually encoded as:

```text
0 -> negative
4 -> positive
```

In the experiments, the positive class is remapped from `4` to `1`.

---

### arXiv Metadata Dataset

The arXiv dataset is a snapshot of scientific-paper metadata containing titles, abstracts, categories, authors, and publication information.

- Text fields used: title and abstract.
- Binary problem:
  - `cs`: the primary category belongs to Computer Science.
  - `no_cs`: all remaining primary categories.
- Balanced sample used in the experiments: **100,000 papers**.
- Characteristics: longer, technical, formal, and semantically dense documents.

#### Download

The metadata snapshot used in this project can be obtained from the Cornell University arXiv dataset on Kaggle:

- Dataset page:  
  <https://www.kaggle.com/datasets/Cornell-University/arxiv>

The relevant file is:

```text
arxiv-metadata-oai-snapshot.json
```

Using the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
```

A Kaggle account and API token may be required. See:

<https://www.kaggle.com/docs/api>

The downloaded snapshot may be updated over time, so the exact number of records can differ from the version used in the experiments.

---

### text8

The **text8** corpus is a cleaned English Wikipedia text collection commonly used to train and evaluate word-embedding models.

In this project, text8 is used to train general-domain **Skip-gram with Negative Sampling** embeddings before comparing them with domain-specific embeddings trained on Sentiment140 or arXiv.

#### Download

- Dataset information page:  
  <http://mattmahoney.net/dc/textdata.html>

- Direct download:  
  <http://mattmahoney.net/dc/text8.zip>

Example:

```bash
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
```

The extracted file is:

```text
text8
```

It contains a single line of preprocessed, space-separated English text.

---

## Suggested Data Directory

A possible local directory structure is:

```text
data/
├── sentiment140/
│   └── training.1600000.processed.noemoticon.csv
├── arxiv/
│   └── arxiv-metadata-oai-snapshot.json
└── text8/
    └── text8
```

When using Google Colab and Google Drive, the notebooks in this repository may need paths similar to:

```text
/content/drive/MyDrive/TFM/Dataset/
```

Update all input and output paths before running the notebooks.

---

## Text Preprocessing

A common preprocessing procedure is applied so that the observed differences primarily depend on the representation rather than on different text-cleaning pipelines.

The workflow includes:

1. lowercasing;
2. URL, HTML tag, and mention removal;
3. whitespace normalization;
4. tokenization with spaCy;
5. punctuation and number removal;
6. stopword removal;
7. explicit preservation of negations;
8. lemmatization;
9. removal of empty documents;
10. inspection of missing values and duplicates.

---

## Experimental Methodology

All representations are evaluated under a common protocol:

1. homogeneous preprocessing;
2. balanced sampling;
3. stratified train-test split;
4. fitting the representation on the training set;
5. transforming the test set;
6. classification with a **Random Forest**;
7. predictive and computational evaluation;
8. interpretability and geometric analysis.

The same classifier is used to isolate the effect of the vector representation.

### Predictive Metrics

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### Computational and Geometric Metrics

- representation construction time;
- transformation time;
- classifier training and prediction time;
- dimensionality;
- sparsity;
- memory usage;
- mean vector norm;
- within-class cosine similarity;
- between-class cosine similarity;
- distance between class centroids;
- entropy of topic distributions.

---

## Main Results

> The reported execution times correspond to the experimental environment used in the project and should not be interpreted as universal benchmarks.

### Sentiment140

| Representation | Dim. | Accuracy | F1 | AUC-ROC | Representation time (s) | Classifier time (s) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Manual TF-IDF | 2435 | 0.7250 | 0.7250 | 0.7949 | 0.92 | 149.02 | 0.9980 |
| Manual LSA | 100 | 0.7163 | 0.7162 | 0.7797 | 51.46 | 22.06 | 0.0144 |
| Manual LDA | 10 | 0.5285 | 0.5280 | 0.5335 | 115.54 | 1.95 | 0.0000 |
| Wikipedia SG-NS + TF-IDF | 100 | 0.6148 | 0.6146 | 0.6681 | 332.28 | 44.83 | 0.0000 |
| Sentiment140 SG-NS + TF-IDF | 100 | 0.6968 | 0.6965 | 0.7624 | 31.85 | 19.89 | 0.0000 |

**Main findings for Sentiment140:**

- TF-IDF achieves the strongest overall performance.
- LSA remains close to TF-IDF while producing a substantially more compact representation.
- LDA struggles with short and noisy tweets.
- SG-NS improves when embeddings are trained on the target domain.
- Direct lexical signals are especially important for sentiment classification.

### arXiv

| Representation | Dim. | Accuracy | F1 | AUC-ROC | Representation time (s) | Classifier time (s) | Sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Manual TF-IDF | 5000 | 0.9200 | 0.9199 | 0.9692 | 1.97 | 49.22 | 0.9878 |
| Manual LSA | 100 | 0.9193 | 0.9192 | 0.9681 | 157.27 | 15.63 | 0.0000 |
| Manual LDA | 10 | 0.9118 | 0.9117 | 0.9647 | 1923.32 | 3.06 | 0.0000 |
| Wikipedia SG-NS + TF-IDF | 100 | 0.8785 | 0.8785 | 0.9444 | 332.28 | 28.10 | 0.0000 |
| arXiv SG-NS + TF-IDF | 100 | 0.9255 | 0.9255 | 0.9730 | 220.15 | 20.41 | 0.0000 |

**Main findings for arXiv:**

- TF-IDF, LSA, and LDA produce competitive results.
- LSA maintains performance close to TF-IDF with much lower dimensionality.
- LDA performs better than on Sentiment140 because scientific abstracts exhibit a clearer topic structure.
- SG-NS trained on arXiv achieves the strongest overall result.
- Domain-specific embeddings outperform general Wikipedia-based embeddings.

---

## Interpretability

The project includes several interpretability strategies.

### TF-IDF

- terms with the largest IDF values;
- class-relevant terms;
- cosine similarity between terms;
- PCA projections;
- similar-document retrieval.

### LSA

- singular values;
- cumulative energy;
- highest-weight terms by latent component;
- class centroids;
- two-dimensional projections;
- approximate reconstruction in the original term space.

### LDA

- most probable words by topic;
- topic distribution for individual documents;
- dominant-topic distribution;
- topic-weight differences between classes;
- topic entropy;
- document-topic and topic-word matrices.

### SG-NS

- nearest words;
- cosine similarity;
- local PCA projections;
- comparison between general and domain-specific embeddings;
- document construction and retrieval using SG-NS + TF-IDF.

---

## Manual and Library Implementations

The repository includes custom implementations of the main methods and comparisons with optimized library versions.

| Method | Manual implementation | Reference implementation |
|---|---|---|
| TF-IDF | Explicit TF, IDF, and document-matrix calculation | `TfidfVectorizer` |
| LSA | Manual TF-IDF + NumPy SVD | `TruncatedSVD` |
| LDA | Collapsed Gibbs sampling | `LatentDirichletAllocation` |
| SG-NS | Pair generation, negative sampling, and neural training | TensorFlow / pretrained embeddings |

The manual versions prioritize correspondence with the mathematical formulation. Library implementations are more efficient because they use sparse matrices, vectorized operations, and optimized algorithms.

---

## Technologies

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- spaCy
- TensorFlow
- Matplotlib
- tqdm
- Google Colab
- Google Drive

The experiments were mainly executed in Google Colab. When available, an **NVIDIA Tesla T4 with 16 GB of memory** was used. However, several stages—such as preprocessing, vocabulary construction, and frequency counting—primarily depend on CPU and system memory.

---

## Installation

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn spacy tensorflow matplotlib tqdm adjustText kaggle
python -m spacy download en_core_web_sm
```

---

## Execution

The recommended workflow is:

1. download or prepare the datasets;
2. update the input and output paths;
3. run the preprocessing notebooks;
4. generate the vector representations;
5. train the classifier;
6. evaluate the metrics;
7. run the interpretability analyses.

Main notebooks developed for the project:

```text
Copia_de_sentiment_140_preprocess.ipynb
Sentiment140_vectors.ipynb
arxiv_representation.ipynb
Skip_Gram_Prueba.ipynb
```

When using Google Colab, review all paths following patterns such as:

```python
/content/drive/MyDrive/TFM/
```

---

## Reproducibility

To support homogeneous comparisons:

- fixed random seeds are used;
- samples are balanced by class;
- train-test splits are stratified;
- the same classifier is reused across representations;
- the same data partitions are maintained whenever possible;
- representation and classification times are recorded separately.

Results may vary depending on:

- hardware;
- library versions;
- corpus size;
- GPU availability;
- dimensionality;
- hyperparameters;
- the version of the arXiv metadata snapshot.

---

## Main Conclusions

1. No representation is universally superior.
2. TF-IDF is a highly competitive baseline for short texts and direct lexical signals.
3. LSA provides a strong balance between predictive performance and dimensionality reduction.
4. LDA is more useful when documents are sufficiently long and topic-coherent.
5. SG-NS strongly depends on the domain used to train the embeddings.
6. Domain-specific embeddings can outperform embeddings trained on general corpora.
7. Efficiency should be evaluated over the complete pipeline, not only during representation construction.
8. Interpretability takes different forms for each method.
9. Manual implementations improve understanding but do not match the efficiency of optimized libraries.

---

## Limitations

- The evaluation focuses on Sentiment140 and arXiv.
- The experimental tasks are simplified to binary classification.
- A single supervised classifier is used.
- The manual implementations prioritize clarity over efficiency.
- Embedding aggregation loses information about word order.
- Two-dimensional PCA projections are exploratory and do not preserve the complete structure of the original space.

---

## Future Work

- evaluation on additional domains and languages;
- multiclass and multilabel classification;
- information retrieval and clustering;
- comparison with RNN, LSTM, and GRU models;
- inclusion of BERT and Transformer architectures;
- contextual embeddings;
- systematic hyperparameter optimization;
- comparison with additional classifiers;
- study of representations that preserve sequential structure.

---

## References

The theoretical framework builds on foundational work including:

- Salton et al. on the vector-space model and TF-IDF.
- Deerwester et al. on Latent Semantic Analysis.
- Blei, Ng, and Jordan on Latent Dirichlet Allocation.
- Mikolov et al. on Skip-gram and Negative Sampling.
- Go, Bhayani, and Huang on Sentiment140.

The complete bibliography is available in the academic document associated with the project.

---

## Academic Use

This repository accompanies an academic project focused on the relationship between:

- mathematical formulation;
- computational implementation;
- predictive performance;
- computational cost;
- dimensionality;
- geometry;
- interpretability.

When reusing code, results, or figures, please cite the project and the original sources of the datasets and algorithms.
