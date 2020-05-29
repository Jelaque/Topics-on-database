import multiprocessing as mp,os

class Chunker(object):
   #Iterator that yields start and end locations of a file chunk of default size 1MB.
   @classmethod
   def chunkify(cls,fname,size=100*1024*1024):
      fileEnd = os.path.getsize(fname)
      with open(fname,'rb') as f:
         chunkEnd = f.tell()
         while True:
            chunkStart = chunkEnd
            f.seek(size,1)
            cls._EOC(f)
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd >= fileEnd:
               break

   #Move file pointer to end of chunk
   @staticmethod
   def _EOC(f):
      f.readline()

   #read chunk
   @staticmethod
   def read(fname,chunk):
      with open(fname,'rb') as f:
         f.seek(chunk[0])
         return f.read(chunk[1])