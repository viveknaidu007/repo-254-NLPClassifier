import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import os
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt  # Matplotlib for plotting
import wordninja

# Function to preprocess text
def preprocess_text(text, stopwords_file):
    text = text.lower()
    tokens = word_tokenize(text)
    
    filtered_tokens = []
    for token in tokens:
        if not token.startswith("http://") and not token.startswith("https://"):
            if re.match("^[^\W\d_]+$", token):
                if len(token) > 10:
                    processed_words = split_words([token])
                    filtered_tokens.extend(processed_words)
                else:
                    filtered_tokens.append(token)
    
    with open(stopwords_file, 'r') as file:
        custom_stopwords = file.read().splitlines()

    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = set(custom_stopwords + list(nltk_stopwords))
    filtered_tokens = [token for token in filtered_tokens if token not in all_stopwords]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

# Function to split words
def split_words(word_list):
    new_word_list = []
    for word in word_list:
        if len(word) > 10:
            split_words = wordninja.split(word)
            new_word_list.extend(split_words)
        else:
            new_word_list.append(word)
    return new_word_list

# Function to classify documents
def classify_documents(k, query):
    directories = [r"C:\Users\user\Documents\6th semester\IR\ResearchPapers"]
    stopwords_file = r"C:\Users\user\Documents\6th semester\IR\Stopword-List.txt"
    total_docs = len(os.listdir(directories[0]))

    documents = []
    labels = []

    for filename in os.listdir(directories[0]):
        try:
            with open(os.path.join(directories[0], filename), 'r') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text, stopwords_file)
                documents.append(preprocessed_text)
                labels.append(assign_labels([filename])[filename])
        except UnicodeDecodeError:
            print(f"Error reading file: {filename}")

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(documents)
    y = np.array(labels)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    query_vector = tfidf_vectorizer.transform([preprocess_text(query, stopwords_file)])
    query_pred = knn.predict(query_vector)[0]

    similarities = np.dot(X, query_vector.T).toarray().reshape(-1)
    relevant_docs_indices = np.argsort(similarities)[::-1][:5]
    relevant_docs = [os.listdir(directories[0])[i] for i in relevant_docs_indices]

    y_pred = knn.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')

    return query_pred, relevant_docs, accuracy, precision, recall, f1

# Function to assign labels
def assign_labels(document_ids):
    labels = {}
    for doc_id in document_ids:
        if doc_id in ['1.txt', '2.txt', '3.txt', '7.txt']:
            labels[doc_id] = "Explainable Artificial Intelligence"
        elif doc_id in ['8.txt', '9.txt', '11.txt']:
            labels[doc_id] = "Heart Failure"
        elif doc_id in ['12.txt', '13.txt', '14.txt', '15.txt', '16.txt']:
            labels[doc_id] = "Time Series Forecasting"
        elif doc_id in ['17.txt', '18.txt', '21.txt']:
            labels[doc_id] = "Transformer Model"
        elif doc_id in ['22.txt', '23.txt', '24.txt', '25.txt', '26.txt']:
            labels[doc_id] = "Feature Selection"
    return labels

# Function to perform clustering
def perform_clustering(num_clusters):
    directories = [r"C:\Users\user\Documents\6th semester\IR\ResearchPapers"]
    stopwords_file = r"C:\Users\user\Documents\6th semester\IR\Stopword-List.txt"
    total_docs = len(os.listdir(directories[0]))

    documents = []

    for filename in os.listdir(directories[0]):
        try:
            with open(os.path.join(directories[0], filename), 'r') as file:
                text = file.read()
                preprocessed_text = preprocess_text(text, stopwords_file)
                documents.append(preprocessed_text)
        except UnicodeDecodeError:
            print(f"Error reading file: {filename}")

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(documents)

    # Reduce dimensionality for visualization
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.5)
    plt.title('Document Clustering')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    st.pyplot(plt)  # Display plot in Streamlit

# Streamlit UI code
def main():
    st.title("Document Classifier and Clustering")

    task = st.sidebar.selectbox("Choose Task", ["Text Classification", "Text Clustering"])

    if task == "Text Classification":
        st.header("Text Classification")
        k = st.number_input("Enter the value of K for KNN", min_value=1, step=1)
        query = st.text_area("Enter the query")

        if st.button("Classify"):
            query_pred, relevant_docs, accuracy, precision, recall, f1 = classify_documents(k, query)
            st.subheader("Classification Results")
            st.info(f"Predicted Category: {query_pred}")
            st.write("Top Relevant Documents:")
            for doc in relevant_docs:
                st.write(doc)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-score: {f1:.2f}")

    elif task == "Text Clustering":
        st.header("Text Clustering")
        num_clusters = st.number_input("Enter the number of clusters", min_value=2, step=1)
        if st.button("Cluster"):
            perform_clustering(num_clusters)  # Call clustering function

    st.sidebar.title("About")
    st.sidebar.info("This app performs text classification and clustering.")

if __name__ == "__main__":
    main()
