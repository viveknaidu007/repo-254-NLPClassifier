# Text Classification and Clustering Tool

## Overview
This Streamlit application integrates advanced text processing techniques including text classification using K-nearest neighbors (KNN) and document clustering using K-means. It preprocesses text data, applies TF-IDF vectorization, and utilizes dimensionality reduction for visualization. Users can classify documents based on input queries and cluster documents for exploratory analysis.

## Features

1. **User Interface**
   - Interactive web interface powered by Streamlit for intuitive user interaction.
   - Supports two main functionalities: Text Classification and Text Clustering.

2. **Text Preprocessing**
   - Tokenization: Splits text into tokens using NLTK's `word_tokenize`.
   - Lowercasing: Converts text to lowercase for uniformity.
   - URL and Punctuation Filtering: Removes URLs and non-alphanumeric characters.
   - Stopword Removal: Utilizes NLTK's stopwords combined with custom stopwords.
   - Word Splitting: Splits long concatenated words using `wordninja`.
   - Stemming: Reduces words to their root forms using Porter Stemmer.

3. **Text Classification**
   - Utilizes K-nearest neighbors (KNN) algorithm to classify documents.
   - Calculates TF-IDF features for text representation.
   - Provides accuracy, precision, recall, and F1-score metrics for evaluation.
   - Identifies top relevant documents based on cosine similarity.

4. **Text Clustering**
   - Implements K-means clustering to group documents into clusters.
   - Visualizes document clusters in 2D space using Truncated SVD for dimensionality reduction.
   - Enables exploration of document relationships and similarities within clusters.

5. **Performance Metrics**
   - Computes classification metrics (accuracy, precision, recall, F1-score) to evaluate model performance.
   - Provides insights into the effectiveness of document classification.

6. **Data Handling**
   - Reads and preprocesses text documents from specified directories.
   - Handles exceptions like UnicodeDecodeError gracefully during file reading.

## Libraries and Imports Used
- **Streamlit**: Front-end web application framework for interactive UI.
- **scikit-learn**: Machine learning library for KNN, K-means, TF-IDF vectorization, and metrics.
- **NLTK**: Natural Language Toolkit for text preprocessing (tokenization, stopwords, stemming).
- **matplotlib**: Visualization library for plotting document clusters.
- **wordninja**: Library for splitting concatenated words.
- **os, re, numpy**: Standard Python libraries for file operations, regular expressions, and numerical operations.

## Applications
1. **Document Management**: Efficiently classify and organize large volumes of text documents.
2. **Exploratory Data Analysis**: Visualize and explore document clusters to identify patterns and insights.
3. **Educational Tools**: Learn about text classification, clustering algorithms, and NLP techniques.

## How to Use
### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```sh
   python -m nltk.downloader punkt stopwords
   ```

### Configuration
- Update directory paths (`directories`, `stopwords_file`) in the `main` function according to your local setup.

### Running the App
1. Launch the app using Streamlit:
   ```sh
   streamlit run text.py
   ```
2. Access the application through the provided local URL (typically `http://localhost:8501`).

### Usage
- Choose between Text Classification or Text Clustering tasks.
- For Text Classification:
  - Input the value of `K` for KNN and a query to classify.
  - View predicted category, top relevant documents, and evaluation metrics.
- For Text Clustering:
  - Specify the number of clusters.
  - Visualize document clusters in 2D space and explore document groupings.

## Conclusion
This advanced text processing tool leverages Streamlit to offer a seamless user experience for document classification and clustering tasks. It combines powerful NLP techniques with interactive visualization, making it ideal for both educational purposes and practical applications in data analysis and document management.

## Acknowledgements
- Streamlit: https://streamlit.io/
- scikit-learn: https://scikit-learn.org/
- NLTK: https://www.nltk.org/
- wordninja: https://github.com/keredson/wordninja
