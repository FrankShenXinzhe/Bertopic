# 中文文本聚类与主题词提取（Bertopic for Chinese）

## 项目简介

本项目旨在通过自然语言处理技术对文本进行聚类分析，并提取每个聚类（主题）下的关键主题词。使用的方法包括句嵌入（通过预训练语言模型）、KMeans/HDBSCAN聚类、TF-IDF主题词提取，以及UMAP降维可视化。

## 主要功能

1. **句嵌入计算**：使用预训练的语言模型（BAAI/bge-large-zh-noinstruct）将文本转换为高维向量。
2. **聚类分析**：支持KMeans和HDBSCAN两种聚类方法，用户可以选择适合的聚类算法。
3. **主题词提取**：对每个聚类（主题）下的评论进行TF-IDF分析，提取最重要的主题词。
4. **可视化**：使用UMAP对句嵌入进行降维，并在二维平面上可视化聚类结果。
5. **结果输出**：输出每个评论的聚类标签、主题词及其重要性，以及每个评论到其聚类中心的距离排序。

## 使用方法

1. **环境准备**：
   - 确保安装了Python及相关库：`umap`, `torch`, `jieba`, `hdbscan`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `transformers`。
   - 可以使用`pip install`命令安装缺失的库。

2. **运行代码**：
   - 直接运行提供的Python脚本。
   - 在KMeans聚类部分，根据手肘法则输出的SSE图，人工判断并输入最佳聚类个数。

3. **结果查看**：
   - 代码运行后，将输出句子-主题对应关系表、每个主题下的重要主题词，以及评论到聚类中心的距离排序。
   - 结果还将保存为`result.xlsx`文件，方便进一步分析。

## 注意事项

- 聚类效果受多种因素影响，包括预训练模型的选择、聚类算法及参数设置等。
- 主题词的提取基于TF-IDF算法，可能需要根据具体应用场景调整参数。
- UMAP可视化提供了直观的聚类结果展示，但具体解释需结合业务背景。


# Bertopic for Chinese

## Project Overview

This project aims to perform clustering analysis on texts through natural language processing techniques and extract key topic words for each cluster (topic). The methods used include sentence embedding (via a pre-trained language model), KMeans/HDBSCAN clustering, TF-IDF topic word extraction, and UMAP dimensionality reduction for visualization.

## Main Features

1. **Sentence Embedding Calculation**: Converts texts into high-dimensional vectors using a pre-trained language model (BAAI/bge-large-zh-noinstruct).
2. **Clustering Analysis**: Supports two clustering methods, KMeans and HDBSCAN, allowing users to choose the appropriate clustering algorithm.
3. **Topic Word Extraction**: Performs TF-IDF analysis on reviews within each cluster (topic) to extract the most important topic words.
4. **Visualization**: Uses UMAP to reduce the dimensionality of sentence embeddings and visualizes the clustering results on a 2D plane.
5. **Result Output**: Outputs the clustering label for each review, topic words and their importance, and a sorted list of reviews based on their distance to the cluster center.

## Usage Instructions

1. **Environment Preparation**:
   - Ensure Python and relevant libraries are installed: `umap`, `torch`, `jieba`, `hdbscan`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `transformers`.
   - Use the `pip install` command to install any missing libraries.

2. **Running the Code**:
   - Directly run the provided Python script.
   - During the KMeans clustering section, manually determine and input the optimal number of clusters based on the SSE plot output by the elbow method.

3. **Viewing Results**:
   - After running the code, it will output a table of sentence-topic correspondences, important topic words for each topic, and a sorted list of reviews based on their distance to the cluster center.
   - The results will also be saved to an `result.xlsx` file for further analysis.

## Notes

- Clustering effectiveness is influenced by various factors, including the choice of pre-trained model, clustering algorithm, and parameter settings.
- Topic word extraction is based on the TF-IDF algorithm and may require parameter adjustments depending on the specific application scenario.
- UMAP visualization provides an intuitive display of clustering results, but interpretation should be combined with business context.

---
