from enum import Enum
from collections import MutableMapping


class Set(Enum):
    DOMAIN = 0
    CODOMAIN = 1

class one2one(MutableMapping):
    x2y = None
    y2x = None
    unknown_val = None

    @staticmethod
    def __keyNotFound():
        return "Key not in domain or codomain."

    def __init__(self, unknown_val):
        self.x2y = {}
        self.y2x = {}

    def __getitem__(self, key):
        if key in self.x2y:
            return self.x2y[key]
        if key in self.y2x:
            return self.y2x[key]
        raise KeyError(one2one.__keyNotFound())

    def __setitem__(self, key, value):
      if key in x2y:
          del self.y2x[ self.x2y[key] ]
      elif key in y2x:
          del self.x2y[ self.y2x[key] ]
      self.x2y[key] = value
      self.y2x[value] = key

    def __delitem__(self, key):
        if key in self.x2y:
            value = self.x2y[key]
            del self.x2y[key]
            del self.y2x[value]
        elif key in self.y2x:
            value = self.y2x[key]
            del self.y2x[key]
            del self.x2y[valuel]
        else:
            raise KeyError(one2one.__keyNotFound())

    def __iter__(self):
        return self.x2y.__iter__() #Is there another way to call this?

    def __len__(self):
        return len(self.x2y)
    



