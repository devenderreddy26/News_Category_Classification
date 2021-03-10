# News_Category_Classification 

## Business Problem
### 1. Description

Source: https://www.kaggle.com/rmisra/news-category-dataset
Data: Huffpost
Download archive.zip from Kaggle.

Problem statement :
This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The task is to train a model to predict the news category of an article

### 2. Machine Learning Problem Formulation

2.1 Data Overview
Source: https://www.kaggle.com/rmisra/news-category-dataset
We have one data file News_Category_Dataset_v2.json which contains columns: 'category' 'headline' 'authors' 'link' 'short_description' 'date'

2.2 Mapping the real-world problem to an ML problem
There are 40+ different categories a particular news can be classified into => Multi class classification problem

### Prerequisites
You must have Scikit Learn, Pandas and other required libraries (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :

1. NewsCategory.ipynb - This contains code fot our Machine Learning model to predict the news category based on training data in 'News_Category_Dataset_v2.json' 
2. NewsCategoryFlask.py - This contains Flask APIs that receives news text through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in NewsCategoryFlask.py and dispalys the news category. (yet to create)
4. templates - This folder contains the HTML template to allow user to enter news text and displays the predicted news category. (yet to create)

### Next Steps
Create the requirements.txt file containing all the required libraries
Create a Docker image and container hosting the model and API with necessary environment setup and start-up operations
Publish the container on to the Amazon container registry (ECR)
Use the Kubernetes service to scale up the docker images as the traffic surges

