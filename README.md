
# phd case study for DTU RECODE

This repository contains the code for the phd case study for DTU RECODE. 

![test image](images_pp/test_image.png)
![similar images](similar_images.png)

## [EDA.py](EDA.py)

This script is used for Exploratory Data Analysis (EDA). It includes:

- Loading the dataset from `styles.csv`
- Performing basic EDA on the styles data, such as distribution of categories and checking for missing values
- Loading and analyzing images from the `images_pp` directory
- Displaying sample images and images with low height
- Analyzing image sizes and plotting the distribution of image widths and heights

## [model.py](model.py)

This script is responsible for defining, training, and evaluating the machine learning model. It includes:

- Defining the architecture of the model
- Compiling the model with the appropriate optimizer and loss function
- Training the model on the preprocessed data
- Evaluating the model's performance on the test data
- Saving the trained model for future use

## [pre-processing.py](pre-processing.py)

This script is used for preprocessing the raw data to make it suitable for machine learning. It includes:

- Cleaning the data by handling missing values and outliers
- Encoding categorical variables
- Normalizing numerical variables
- Splitting the data into training and test sets
- Saving the preprocessed data for future use