# Generalized Cross-Multi Domains Representations using User Sentiments
This project is part of Information Retrieval - Winter 2023 course.

<hr/>

# Project Description
The main objective of the project is to carry out the task of generating generalized multi-cross-domains representations while taking into account user reviews and sentiments which would be useful in a cross-domain recommendation system. Since most of the cross-domain work has been done across two domains. Thus, in the proposed task, given an adjacency list T of items (from multiple K domains where K >= 2) and common users across k domains, along with their metadata

# Dataset
The Amazon review dataset is a collection of user reviews and ratings for products sold on Amazon. It contains product ID, user ID, review text, rating, and other metadata. It's commonly used in cross-domain recommendation systems as it provides a diverse set of products from various categories. It's a valuable resource for researchers and data scientists in the field of recommendation systems. Dataset Link: https://jmcauley.ucsd.edu/data/amazon/

![image](https://user-images.githubusercontent.com/55682564/233842830-0f64b9ea-4d6b-4411-ac0a-fe68dc5c4fbd.png)

# Evaluation Metrics

![image](https://user-images.githubusercontent.com/55682564/233843455-9ac1e48b-66ee-4a9d-95ff-cbc6d6be5321.png)


# Methology

## K-Domain Cross Domain Recommendation

### Data Preprocessing
Initially, we had datasets for three domains, i.e., cell phones, digital music, and movies.

Our aim was first to find the common users who had given ratings for products in each domain and then create datasets for each domain that had the common users and the items in that domain they interacted with, respectively.

To achieve our aim, we first removed the tuples of the products with less than ten ratings from each data set. Then we found the common users by finding the common UserIDs that existed in the datasets of all domains. After all the iterations, we found 681 common users from all the domains.

Now to generate the intersection datasets, we, one by one, picked the dataset of each domain and found the intersection of it with the common users' data based on ProductID. 


### Generalised Loss function
![image](https://user-images.githubusercontent.com/55682564/233843124-739835fd-9415-49c1-a3ee-8010bcd9c964.png)

### Command to run the code
python -u .\train_rec.py --num_epoch 50 --k 3 --dropout 0.5


## Sentiment-Based Cross Domain Recommendation


### Data Preprocessing

As part of the Sentiment-based CDR model we have consider 2 domains - Cellphones and Digital music of the 5-core Amazon Review Dataset.

Each entry had a user-item pair with a corresponding review and customer rating.

Thus we find three types of sentiments scores - 1(base case), VADER-sentiments and customer ratings.

The VADER-based sentiments are found using VADER compound score of the customer review and is then scaled to 0 to 1. To apply the VADER model we apply basic text preprocessing techniques like punctuation removal, stopwords removal, stemming.

Similarly the customer ratings are scaled to the range 0 to 1 from 0 to 5.


### Loss Function
![image](https://user-images.githubusercontent.com/55682564/233843380-b239d0df-ff16-4a40-9ba7-f5811c0f3398.png)

### Command to run the code

python -u .\train_rec.py --num_epoch 50 --dataset phones_music --sentiment sentiment_scaled


# Libraries Required
PyTorch

Scipy

Numpy

math

tensorflow

