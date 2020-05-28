import ml100k
import time as tm

r = ml100k.recommenderMl100k(0)
total_time = 0
for i in range(10):
  start_time = tm.time()
  r.loadMovieLensParallel('../datasets/ml-100k/')
  total_time += tm.time() - start_time
print(total_time/10)
total_time = 0
for i in range(10):
  start_time = tm.time()
  r.loadMovieLens('../datasets/ml-100k/')
  total_time += tm.time() - start_time
print(total_time/10)
