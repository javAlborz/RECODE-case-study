from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('styles.csv')

#Styles EDA
print(df.head())
print(df.describe())

# Distribution of categories
for column in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']:
    print(f"Distribution of {column}:")
    print(df[column].value_counts())
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=90)
    plt.title(f'Distribution of {column}')
    plt.show()

# Check for missing values
print(df.isnull().sum())



#Image EDA
image_dir = 'images_pp'

# Function to display a sample of images
def display_sample_images(df, image_dir, n=5):
    sample_images = df.sample(n)
    plt.figure(figsize=(12, 10))
    for index, row in enumerate(sample_images.itertuples(), 1):
        image_path = os.path.join(image_dir, str(getattr(row, 'id')) + '.jpg') # Add your image extension here
        image = Image.open(image_path)
        plt.subplot(1, n, index)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

display_sample_images(df, image_dir)

# Analyzing image sizes
image_sizes = []
for image_name in df['id']:
    image_path = os.path.join(image_dir, str(image_name) + ".jpg")
    with Image.open(image_path) as img:
        image_sizes.append(img.size)

# Convert to a DataFrame for analysis
image_sizes_df = pd.DataFrame(image_sizes, columns=['Width', 'Height'])

# Plotting the distribution of widths and heights
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(image_sizes_df['Width'], kde=True)
plt.title('Distribution of Image Widths')

plt.subplot(1, 2, 2)
sns.histplot(image_sizes_df['Height'], kde=True)
plt.title('Distribution of Image Heights')
plt.show()


low_height_images = image_sizes_df[image_sizes_df['Height'] < 65.5]

print("Images with height less than 62.5:", df.loc[low_height_images.index, 'id'])

def display_low_height_images(df, image_dir, ids):
    plt.figure(figsize=(12, 10))
    for index, image_id in enumerate(ids, 1):
        image_path = os.path.join(image_dir, str(image_id) + '.jpg')
        if os.path.exists(image_path):
            image = Image.open(image_path)
            plt.subplot(1, len(ids), index)
            plt.imshow(image)
            plt.axis('off')
        else:
            print(f"File not found: {image_path}")
    plt.show()

low_height_image_ids = df.loc[low_height_images.index, 'id']
display_low_height_images(df, image_dir, low_height_image_ids)

# Summary statistics for image sizes
print("Summary statistics for image widths and heights:")
print(image_sizes_df.describe())

# Distribution of unique image widths
print("\nDistribution of unique image widths:")
print(image_sizes_df['Width'].value_counts())

# Distribution of unique image heights
print("\nDistribution of unique image heights:")
print(image_sizes_df['Height'].value_counts())
