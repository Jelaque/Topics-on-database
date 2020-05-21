import math as mt
import pandas as pd
import dict2csv as d2csv

class Recommender:
  def __init__(self,filename):
    self.filename = filename
    self.data = {}

  def read_csv(self):

    data = pd.read_csv(self.filename,index_col=0)
    #data = data.T
    data = data.fillna(0)
    self.data = data.to_dict()

    return self.data

def manhattan(rating1, rating2):
  """Simplest distance operation using only sums"""

  distance = 0
  for key in rating1:
    if key in rating2:
      distance += abs(rating1[key] - rating2[key])
  return distance

def euclidian(rating1, rating2):
  """Usual triangular distance"""

  distance = 0
  for key in rating1:
    if key in rating2:
      distance += pow(rating1[key] - rating2[key],2)
  return mt.sqrt(distance)

def minkowski(rating1, rating2, p):
  """Generalization of manhattan and euclidian distances"""

  distance = 0
  for key in rating1:
    if key in rating2:
      distance += pow(abs(rating1[key] - rating2[key]),p)
  if distance != 0:
    return pow(distance, 1/p)
  return 0

"""Database of rated movies"""
movies = Recommender('../datasets/Movie_Ratings.csv')
movies = movies.read_csv()

"""Database of rated bands of music"""
#bands = d2csv.music_bands
bands = d2csv.users

print(euclidian(bands['Chan'],bands['Veronica']))
print(manhattan(bands['Angelica'],bands['Dan']))
print(minkowski(bands['Jordyn'],bands['Hailey'],1))
print(minkowski(bands['Jordyn'],bands['Hailey'],2))

print(minkowski(movies['Katherine'],movies['Chris'],1))
print(minkowski(movies['Katherine'],movies['Chris'],2))

print(minkowski(movies['Jessica'],movies['greg'],1))
print(minkowski(movies['Jessica'],movies['greg'],2))

print(minkowski(movies['Zak'],movies['Heather'],1))
print(minkowski(movies['Zak'],movies['Heather'],2))

print(minkowski(movies['Matt'],movies['Valerie'],1))
print(minkowski(movies['Matt'],movies['Valerie'],2))

print(minkowski(movies['Patrick C'],movies['Jeff'],1))
print(minkowski(movies['Patrick C'],movies['Jeff'],2))
