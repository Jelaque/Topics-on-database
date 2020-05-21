import math as mt
import pandas as pd


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

def euclidian(rating1, ranting2):
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

movies = Recommender('Movie_Ratings.csv')
data = movies.read_csv()

print(manhattan(data['Heather'],data['ben']))
