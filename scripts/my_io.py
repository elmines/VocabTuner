import os

def abs_path(path, currFile):
   return os.path.abspath( os.path.join(  os.path.dirname(os.path.realpath(currFile)), path ) )

