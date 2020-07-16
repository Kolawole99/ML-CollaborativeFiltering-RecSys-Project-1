#==================================IMPORTING NEEDED PACKAGES AND LIBRARIES=============================
import pandas as pd
from math import sqrt
import numpy as np


#===============================================DATA============================================

#===================================Working with the Movies dataframe============================
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('data/movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('data/ratings.csv')
movies = movies_df.head()
ratings = ratings_df.head()
print(movies)
print(ratings)

#================Remove the year from the title and giving year it own column================
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
edited_movies = movies_df.head()
print(edited_movies)

#==============================Dropping the genres column=================================
movies_df = movies_df.drop('genres', 1)
final_m = movies_df.head()
print(final_m)

#================================Working with the ratings dataframe==============================
#============================Removes the timestamps from the dataframe========================
ratings_df = ratings_df.drop('timestamp', 1)
final_r = ratings_df.head()
print(final_r)


#===========================COLLABORATVE FILTERING/User-User Filtering==================================

#We will be making use of the Pearson Correlation Function to build the recommender system here.
# The process for creating a User Based recommendation system is as follows:
    # Select a user with the movies the user has watched
    # Based on his rating to movies, find the top X neighbours
    # Get the watched movie record of the user for each neighbour.
    # Calculate a similarity score using some formula
    # Recommend the items with the highest score

#=======================Create an inpur user to recommend movies to==============================
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
        ] 
inputMovies = pd.DataFrame(userInput)
inputMovies

#=========================Add movie id to the user input=============================
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

#========Filtering out users that have watched movies that the input has watched and storing it==========
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

#===========Let look at one of the user with the specified ID======================
specified_user = userSubsetGroup.get_group(1130)
print(specified_user)

#==========Sorting it so users with movie most in common with the input will have priority==============
#Let's also sort these groups so the users that share the most movies in common with the input have higher priority. This provides a richer recommendation since we won't go through every single user
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
#lets look at the first user
userSubsetGroup[0:3]

#========================Finding ther similarity of users same as the input user========================
# We're going to find out how similar each user is to the input through the Pearson Correlation Coefficient. It is used to measure the strength of a linear association between two variables. The formula is finding this coefficient between sets X and Y with N values.
#The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation.

#select a subset of users to iterate through
userSubsetGroup = userSubsetGroup[0:100]

#===============================Calculating the pearson correlation=============================
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonCF = pearsonDF.head()
print(pearsonCF)

#===============================Getting the top 50 similar users================================
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topusers = topUsers.head()
print(topusers)

#=============================Rating of selected users to all movies==============================

# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

#Now let's sort it and see the top 20 movies that the algorithm recommended!
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score',
                    ascending=False)
recommendation_df.head(10) 

recommendation = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
print(recommendation)

# Advantages and Disadvantages of Collaborative Filtering

# Advantages
# Takes other user's ratings into consideration
# Doesn't need to study or extract information from the recommended item
# Adapts to the user's interests which might change over time

# Disadvantages
# Approximation function can be slow
# There might be a low of amount of users to approximate
# Privacy issues when trying to learn the user's preferences