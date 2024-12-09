# Final Project: Book Data Analysis and Genre Classification

This project focuses on two tasks: optimizing SQL query performance for a book metadata dataset (Task 1) and classifying book genres using deep learning and cover images (Task 2).

---

## Folder Structure

Final_Project/
│
├── Task1_Optimization.ipynb      # Jupyter notebook for Task 1
├── Task2_GenreClassification.py  # Python script for Task 2
├── data/                         # Data folder containing CSV files
│   ├── book30-listing-train.csv
│   ├── book30-listing-test.csv
│   ├── bookcover30-labels-train.txt
│   ├── bookcover30-labels-test.txt
└── README.md                     # Project documentation

## Task 1: Query Optimization

**File:** Task1_Optimization.ipynb

### Overview
Task 1 explores SQL query optimization for a structured dataset of best-selling books. The dataset includes metadata such as author names, titles, and genres. The genre column, however, contains missing or inconsistent data. This task focuses on optimizing two key queries:

1. Identifying authors who wrote books in both "Children's" and "Fiction" genres.
2. Finding authors who wrote books spanning all genres authored by J.K. Rowling.

### Dataset Description
Input: A best-selling books dataset (~175 titles) loaded into a SQLite database.
Output: Optimized SQL queries executed on the dataset.

### Optimization Techniques:
- Use of indices on frequently queried columns (Author and Genre).
- Replacement of correlated subqueries with joins.
- Materialized views for repeated subqueries.
- Execution plan analysis with SQLite’s EXPLAIN QUERY PLAN.

### How to Run
Open the Task1_Optimization.ipynb notebook in Jupyter Notebook or JupyterLab and follow the cells to explore the dataset and query optimizations.

---

## Task 2: Genre Classification with Deep Learning

**File:** Task2_GenreClassification.py

### Overview
Task 2 utilizes the BookCover30 dataset to classify book genres based on cover images. Using a pre-trained ResNet50 model fine-tuned with transfer learning, this task demonstrates the potential of deep learning for visual data classification.

### Dataset Description
Training Data: book30-listing-train.csv, bookcover30-labels-train.txt
Testing Data: book30-listing-test.csv, bookcover30-labels-test.txt

Number of Genres: 30
Data Folder: Place all dataset files in the data/ folder within the Final_Project directory.

### How to Run
To execute Task 2, run the script using the following command:

```bash
python Task2_GenreClassification.py output
```


This will:

1. Download approximately 1,700 images from the training and testing datasets.
2. Preprocess images (data augmentation with rotation, cropping, and color jittering).
3. Train the model and evaluate its performance.

**Warning: The script is configured to download approximately 1,700 images by default. Ensure you have a stable internet connection and sufficient storage space. Adjust the NUM_DOWNLOADS_TRAIN and NUM_DOWNLOADS_TEST constants in the script if you wish to limit the downloads.**

### Output
The script generates:

A trained model for genre classification.
Accuracy metrics and confusion matrix visualizations.

### Disclaimer
**Execution times for Task 1 may vary depending on your hardware and system configuration. For Task 2, ensure you have adequate system resources to download and process the images. For large datasets or extended experiments, consider using a high-performance computing platform like Great Lakes.**
