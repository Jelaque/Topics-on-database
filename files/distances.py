import math as mt
import pandas as pd
import dict2csv as d2csv
from chunker import Chunker
import multiprocessing as mp,os

class recommender:
  def __init__(self, data, k=3, metric='pearson', n=5):
    self.k = k
    self.n = n
    self.path = '../datasets'

    self.frequencies = {}
    self.deviations = {}
    self.productid2name = {}

    if type(data).__name__ == 'dict':
      self.data = data

    self.metric = metric
    if self.metric == 'pearson':
      self.fn = self.pearson
    elif self.metric == 'euclidian':
      self.fn = self.euclidian
    elif self.metric == 'cosine':
      self.fn = self.cosine
    elif self.metric == 'manhattan':
      self.fn = self.manhattan

  def convertProductID2name(self, id):
    """Given product id number return product name"""
    if id in self.productid2name:
      return self.productid2name[id]
    else:
      return id

  def readBooks(self):

    self.data = pd.read_csv(self.path+'/ml-books/Movie_Ratings.csv',index_col=0)
    self.data = self.data.fillna(0)
    self.data = self.data.to_dict()

  def readBands(self):

    data = pd.DataFrame.from_dict(d2csv.users, orient='index')
    self.data = (data.T).fillna(0)
    self.data = self.data.to_dict()

  def process_data(self,fname,chunkStart, chunkSize):
    with open(fname, encoding="UTF-8") as f:
      f.seek(chunkStart)
      if (chunkStart == 0):
        f.readline()
      lines = f.read(chunkSize).splitlines()
      for line in lines:
        fields = line.split(',')
        user = fields[0]
        movie = fields[1]
        rating = fields[2]
        timestamp = fields[3]
        if user in self.data:
          currentRatings = self.data[user]
        else:
          currentRatings = {}
        currentRatings[movie] = float(rating)
        self.data[user] = currentRatings

  def readMovies(self, path='/ml-25m'):
    self.data = {}

    #init objects
    pool = mp.Pool(8)
    jobs = []

    #create jobs
    fname = self.path+path+"/ratings.csv"
    for chunkStart,chunkSize in Chunker.chunkify(fname):
      jobs.append( pool.apply_async(self.process_data,(fname,chunkStart,chunkSize)) )

    for job in jobs:
      job.get()
    #clean up
    pool.close()

  def manhattan(self, rating1, rating2):
    """Simplest distance operation using only sums"""

    distance = 0
    for key in rating1:
      if key in rating2:
        distance += abs(rating1[key] - rating2[key])
    return distance

  def euclidian(self, rating1, rating2):
    """Usual triangular distance"""

    distance = 0
    for key in rating1:
      if key in rating2:
        distance += pow(rating1[key] - rating2[key],2)
    return mt.sqrt(distance)

  def minkowski(self, rating1, rating2, p):
    """Generalization of manhattan and euclidian distances"""

    distance = 0
    for key in rating1:
      if key in rating2:
        distance += pow(abs(rating1[key] - rating2[key]),p)
    if distance != 0:
      return pow(distance, 1/p)
    return 0

  def pearson(self, rating1, rating2):
    """Similarity between two users"""

    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
      if key in rating2:
        n += 1
        x = rating1[key]
        y = rating2[key]
        sum_xy += x * y
        sum_x += x
        sum_y += y
        sum_x2 += pow(x, 2)
        sum_y2 += pow(y, 2)
    if n == 0:
      return 0
    # now compute denominator
    denominator = mt.sqrt(sum_x2 - pow(sum_x, 2) / n) * \
                  mt.sqrt(sum_y2 - pow(sum_y, 2) / n)
    if denominator == 0:
      return 0
    else:
      return (sum_xy - (sum_x * sum_y) / n) / denominator

  def cosine(self, rating1, rating2):
    """Similarity for sparse ratings"""

    prod_xy = 0
    len_vx = 0
    len_vy = 0

    for key in rating1:
      if key in rating2:
        prod_xy += rating1[key] * rating2[key]
        len_vx += rating1[key] * rating1[key]
        len_vy += rating2[key] * rating2[key]

    return prod_xy / (mt.sqrt(len_vx) * mt.sqrt(len_vy))

  def computeNearestNeighbor(self, username):
    """creates a sorted list of users based on their distance
    to username"""
    distances = []
    for instance in self.data:
      if instance != username:
        distance = self.fn(self.data[username],self.data[instance])
        distances.append((instance, distance))
    
    distances.sort(key=lambda artistTuple: artistTuple[1], reverse=True)
    return distances

  def recommend(self, user):
    """Give list of recommendations"""
    recommendations = {}
    # first get list of users  ordered by nearness
    nearest = self.computeNearestNeighbor(user)
      #
      # now get the ratings for the user
      #
    userRatings = self.data[user]
      #
      # determine the total distance
    totalDistance = 0.0
    for i in range(self.k):
      totalDistance += nearest[i][1]
      # now iterate through the k nearest neighbors
      # accumulating their ratings
    for i in range(self.k):
      # compute slice of pie 
      weight = nearest[i][1] / totalDistance
      # get the name of the person
      name = nearest[i][0]
      # get the ratings for this person
      neighborRatings = self.data[name]
      # get the name of the person
      # now find bands neighbor rated that user didn't
      for artist in neighborRatings:
        if not artist in userRatings:
          if artist not in recommendations:
            recommendations[artist] = neighborRatings[artist] * \
                                      weight
        else:
          recommendations[artist] = recommendations[artist] + \
                                    neighborRatings[artist] * \
                                    weight
      # now make list from dictionary and only get the first n items
    recommendations = list(recommendations.items())[:self.n]
    recommendations = [(self.convertProductID2name(k), v) for (k, v) in recommendations]
      # finally sort and return
    recommendations.sort(key=lambda artistTuple: artistTuple[1],
                           reverse = True)
    return recommendations[:self.n]

  def ProjectedRanting(self, user, key):

    # first get list of users  ordered by nearness
    nearest = self.computeNearestNeighbor(user)
    projtRating = 0
      #
      # determine the total distance
    totalDistance = 0.0
    for i in range(self.k):
         totalDistance += nearest[i][1]
      # now iterate through the k nearest neighbors
      # accumulating their ratings
    for i in range(self.k):
      # compute slice of pie 
      weight = nearest[i][1] / totalDistance

      name = nearest[i][0]
      projtRating += weight * self.data[name][key]
         
    return projtRating

  def jaccard(self, rating1, rating2):
    """Similarity of number of coincidences between ratings"""
    """returning similarity and distance"""

    union = set()
    for i in rating1.keys():
      union.add(i)
    for i in rating2.keys():
      union.add(i)


    dif = 0
    same = len(union)

    for key in union:
      if (key in rating1) and (key in rating2):
        dif += 1
    
    simil = dif/same
    dist = 1-simil

    return simil,dist