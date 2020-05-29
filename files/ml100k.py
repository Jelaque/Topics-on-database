import codecs
import multiprocessing as mp,os
import math as mt
import pandas as pd
from chunker import Chunker

class recommenderMl100k:

   def __init__(self, data, k=1, metric='pearson', n=5, cores=8):
      
      self.k = k
      self.n = n
      self.username2id = {}
      self.userid2name = {}
      self.productid2name = {}
      self.cores = cores
      self.path = '../datasets/ml-100k/'

      # for some reason I want to save the name of the metric
      self.metric = metric
      if self.metric == 'pearson':
         self.fn = self.pearson
      #
      # if data is dictionary set recommender data to it
      #
      if type(data).__name__ == 'dict':
         self.data = data
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


   def userRatings(self, id, n):
      """Return n top ratings for user with id"""
      print ("Ratings for " + self.userid2name[id])
      ratings = self.data[id]
      print(len(ratings))
      ratings = list(ratings.items())[:n]
      ratings = [(self.convertProductID2name(k), v) \
                 for (k, v) in ratings]
      # finally sort and return
      ratings.sort(key=lambda artistTuple: artistTuple[1],reverse = True)
      for rating in ratings:
         print("%s\t%i" % (rating[0], rating[1]))


   def showUserTopItems(self, user, n):
      """ show top n items for user"""
      items = list(self.data[user].items())
      items.sort(key=lambda itemTuple: itemTuple[1], reverse=True)
      for i in range(n):
         print("%s\t%i" % (self.convertProductID2name(items[i][0]),items[i][1]))


   def process_data(self,fname,chunkStart, chunkSize):
      with open(fname, encoding="ascii", errors="surrogateescape") as f:
         f.seek(chunkStart)
         lines = f.read(chunkSize).splitlines()
         for line in lines:
            fields = line.split('\t')
            user = fields[0]
            movie = fields[1]
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
               currentRatings = self.data[user]
            else:
               currentRatings = {}
            currentRatings[movie] = rating
            self.data[user] = currentRatings

   def process_item(self,fname,chunkStart, chunkSize):
      with open(fname, encoding="ISO-8859-1") as f:
         f.seek(chunkStart)
         lines = f.read(chunkSize).splitlines()
         for line in lines:
            fields = line.split('|')
            mid = fields[0].strip()
            title = fields[1].strip()
            self.productid2name[mid] = title

   def loadBookDB(self, path=''):
        """loads the BX book dataset. Path is where the BX files are
        located"""
        self.data = {}
        i = 0
        #
        # First load book ratings into self.data
        #
        f = codecs.open(path + "BX-Book-Ratings.csv", 'r', 'utf8')
        for line in f:
            i += 1
            #separate line into fields
            fields = line.split(';')
            user = fields[0].strip('"')
            book = fields[1].strip('"')
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
                currentRatings = self.data[user]
            else:
                currentRatings = {}
            currentRatings[book] = rating
            self.data[user] = currentRatings
        f.close()
        #
        # Now load books into self.productid2name
        # Books contains isbn, title, and author among other fields
        #
        f = codecs.open(path + "BX-Books.csv", 'r', 'utf8')
        for line in f:
            i += 1
            #separate line into fields
            fields = line.split(';')
            isbn = fields[0].strip('"')
            title = fields[1].strip('"')
            author = fields[2].strip().strip('"')
            title = title + ' by ' + author
            self.productid2name[isbn] = title
        f.close()
        #
        #  Now load user info into both self.userid2name and
        #  self.username2id
        #
        f = codecs.open(path + "BX-Users.csv", 'r', 'utf8')
        for line in f:
            i += 1
            #print(line)
            #separate line into fields
            fields = line.split(';')
            userid = fields[0].strip('"')
            location = fields[1].strip('"')
            if len(fields) > 3:
                age = fields[2].strip().strip('"')
            else:
                age = 'NULL'
            if age != 'NULL':
                value = location + '  (age: ' + age + ')'
            else:
                value = location
            self.userid2name[userid] = value
            self.username2id[location] = userid
        f.close()

   def process_users(self,fname,chunkStart, chunkSize):
      with open(fname, encoding="UTF-8") as f:
         f.seek(chunkStart)
         lines = f.read(chunkSize).splitlines()
         for line in lines:
            fields = line.split('|')
            userid = fields[0].strip('"')
            self.userid2name[userid] = line
            self.username2id[line] = userid

   def loadMovieLensParallel(self):
      self.data = {}

      #init objects
      pool = mp.Pool(self.cores)
      jobs = []

      #create jobs
      fname = self.path+"u.data"
      for chunkStart,chunkSize in Chunker.chunkify(fname):
         jobs.append( pool.apply_async(self.process_data,(fname,chunkStart,chunkSize)) )

      #wait for all jobs to finish
      for job in jobs:
         job.get()
      #clean up
      pool.close()

      #init objects
      pool = mp.Pool(self.cores)
      jobs = []
      fname = self.path+"u.item"
      for chunkStart,chunkSize in Chunker.chunkify(fname):
         jobs.append( pool.apply_async(self.process_item,(fname,chunkStart,chunkSize)) )

      #wait for all jobs to finish
      for job in jobs:
         job.get()
      #clean up
      pool.close()

      #init objects
      pool = mp.Pool(self.cores)
      jobs = []
      fname = self.path+"u.user"
      for chunkStart,chunkSize in Chunker.chunkify(fname):
         jobs.append( pool.apply_async(self.process_users,(fname,chunkStart,chunkSize)) )

      #wait for all jobs to finish
      for job in jobs:
         job.get()

      #clean up
      pool.close()


   def loadMovieLens(self, path=''):
      self.data = {}
      # first load movie ratings
      i = 0
      # First load book ratings into self.data
      #f = codecs.open(path + "u.data", 'r', 'utf8')
      f = codecs.open(path + "u.data", 'r', 'ascii')
      #  f = open(path + "u.data")
      for line in f:
         i += 1
         #separate line into fields
         fields = line.split('\t')
         user = fields[0]
         movie = fields[1]
         rating = int(fields[2].strip().strip('"'))
         if user in self.data:
            currentRatings = self.data[user]
         else:
            currentRatings = {}
         currentRatings[movie] = rating
         self.data[user] = currentRatings
      f.close()
      #
      # Now load movie into self.productid2name
      # the file u.item contains movie id, title, release date among
      # other fields
      #
      #f = codecs.open(path + "u.item", 'r', 'utf8')
      f = codecs.open(path + "u.item", 'r', 'iso8859-1', 'ignore')
      #f = open(path + "u.item")
      for line in f:
         i += 1
         #separate line into fields
         fields = line.split('|')
         mid = fields[0].strip()
         title = fields[1].strip()
         self.productid2name[mid] = title
      f.close()
      #
      #  Now load user info into both self.userid2name
      #  and self.username2id
      #
      #f = codecs.open(path + "u.user", 'r', 'utf8')
      f = open(path + "u.user")
      for line in f:
         i += 1
         fields = line.split('|')
         userid = fields[0].strip('"')
         self.userid2name[userid] = line
         self.username2id[line] = userid
      f.close()

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

      dem = mt.sqrt(len_vx) * mt.sqrt(len_vy)
      if dem != 0:
         return prod_xy / dem
      return -1

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


   def computeNearestNeighbor(self, username):
      """creates a sorted list of users based on their distance
      to username"""
      distances = []
      for instance in self.data:
         if instance != username:
            distance = self.fn(self.data[username],
                               self.data[instance])
            distances.append((instance, distance))
      # sort based on distance -- closest first
      distances.sort(key=lambda artistTuple: artistTuple[1],
                     reverse=True)
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
      recommendations = [(self.convertProductID2name(k), v)
                         for (k, v) in recommendations]
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