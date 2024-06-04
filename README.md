# Sentiment Analysis Using Opinion Mining

## Project Overview

This project, titled "Sentiment Analysis Using Opinion Mining," was conducted by Aparna Jha, Prachiti Akre, Rushikesh Malu, and Yash Tekade at Shri Ramdeobaba College of Engineering & Management, Nagpur. The project aims to analyze user sentiments from product reviews on e-commerce platforms using machine learning techniques, specifically the XGBoost algorithm.

## Objectives

- Generate a novel dataset using the Affin algorithm and classify reviews based on emotions: Happy, Satisfied, Mixed, and Disappointed.
- Evaluate the collective sentiment expressed by users in their product reviews and assign a rating based on this analysis.
- Automate the labor-intensive process of manually analyzing and contrasting numerous user feedbacks on products.
- Condense all reviews for a specific product into a single numerical score, facilitating comparisons and product recommendations.
- Offer visual representations, such as donut charts and bar graphs, for contrasting various emotions associated with each individual product.

## Methodology

### Data Collection
- **Web Scraping**: Product reviews were collected from e-commerce platforms using web scraping techniques with tools like Beautiful Soup.

### Data Preparation
- **AFFIN Algorithm**: Used to categorize Amazon product reviews into four emotional classes: Happy, Satisfied, Mixed, and Disappointed.

### Data Preprocessing
- **Tokenization**: Breaking down text into discrete words or tokens.
- **Stop Words Removal**: Removing common words that do not contribute to sentiment analysis.
- **Punctuation and Special Characters Removal**: Eliminating irrelevant characters.
- **Lowercasing**: Converting all words to lowercase to ensure consistency.
- **Stemming or Lemmatization**: Reducing words to their base forms.

### Data Classification
- **Algorithm Selection**: XGBoost was chosen for its efficiency and robustness.
- **Model Training**: The model was trained on a labeled dataset and evaluated using metrics like accuracy, precision, recall, and F1 score.

### Data Visualization
- **Donut Graph**: Provides an overview of the emotional distribution of product reviews.
- **Bar Graph**: Allows for a detailed comparison of the product's emotional performance against its competitors.

### UI Development
- **Bootstrap Framework**: Used to create a responsive and visually appealing user interface.
- **Key Features**: Sentiment overview, emotional distribution visualization, and comparative analysis.

## Technology Stack

### For Research Work
- **Python**: General-purpose programming language.
- **Beautiful Soup**: Python package for parsing HTML and XML documents.
- **Scikit-Learn**: Machine learning library for Python.
- **NLTK**: Natural Language Toolkit for symbolic and statistical natural language processing.
- **Pandas**: Data manipulation and analysis library.
- **Requests**: Library for making HTTP requests.

### For Development and Deployment
- **HTML, CSS, JS**: Used for web designing and development.
- **Django**: High-level Python web framework for rapid development and clean design.

## Future Scope
- Implement the model on multiple websites for product comparison.
- Extend the model to take input in various formats such as images and audio.
- Convert the project into a web extension for easy use and convenience.

## Conclusion
This project presents a comprehensive study on sentiment analysis for product reviews. It proposes a machine learning model that effectively analyzes sentiment and allows users to sort products based on keywords. The project demonstrates the accuracy and effectiveness of the model and provides a user-friendly interface for consumers.

## References
- [Sentiment Analysis Using Product Review Data](https://doi.org/10.1186/s40537-015-0015-2)
- [Optimization of Sentiment Analysis Using Machine Learning Classifiers](https://doi.org/10.1186/s13673-017-0116-3)
- [Sentiment Analysis in Amazon Reviews Using Probabilistic Machine Learning](https://www.swarthmore.edu)

## Screenshots
![image](https://github.com/Prachiti15/sentimentAnalysis/assets/91412980/0b688848-4530-46f9-a7bc-22faafee81cc)
![image](https://github.com/Prachiti15/sentimentAnalysis/assets/91412980/a301e683-c1cf-4880-a0b1-e62a0465e435)
![image](https://github.com/Prachiti15/sentimentAnalysis/assets/91412980/126da764-54f1-42be-accc-c3c4a1067520)
