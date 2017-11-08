from enum import Enum
from collections import MutableMapping
from copy import deepcopy

class Set(Enum):
    DOMAIN = 0
    CODOMAIN = 1

class one2one(MutableMapping):
    x2y = None
    y2x = None
    x_type = None
    y_type = None
    unknown_x = None
    unknown_y = None

    @staticmethod
    def __key_not_found(key):
        return str(key) + " not in domain or codomain."

    @staticmethod
    def __breaks_mapping():
        return "Cannot alter unknown_x <--> unknown_y mapping."

    def __bad_type(self):
        return "Key must be of type " + str(self.x_type) + " or " + str(self.y_type) + "."

    def __init__(self, x_type, y_type, unknown_x = None, unknown_y = None):
        self.x2y = {}
        self.y2x = {}
        self.x_type = x_type
        self.y_type = y_type

        if unknown_x and unknown_y:
            self.unknown_x = unknown_x
            self.unknown_y = unknown_y
            self.x2y[unknown_x] = unknown_y
            self.y2x[unknown_y] = unknown_x
        elif unknown_x or unknown_y:
            raise ValueError("One must specify both unknown_x and unknown_y or neither.")

    def __getitem__(self, key):
        if key in self.x2y:
            return self.x2y[key]
        if key in self.y2x:
            return self.y2x[key]
        if self.unknown_x:
            if type(key) == self.x_type: return self.unknown_y
            elif type(key) == self.y_type: return self.unknown_x
            return self.unknown_x

        raise KeyError(one2one.__key_not_found(key))

    def __setitem__(self, key, value):
      #Checked, and "in" does indeed perform a deep equality comparison
      if self.unknown_x in (key, value) or self.unknown_y in (key, value):
           raise ValueError(one2one.__breaks_mapping())
      if type(key) != self.x_type and type(key) != self.y_type:
           raise TypeError(self.__bad_type())

      if key in self.x2y:
          del self.y2x[ self.x2y[key] ]
      elif key in self.y2x:
          del self.x2y[ self.y2x[key] ]
      self.x2y[key] = value
      self.y2x[value] = key

    def __delitem__(self, key):
        if key == self.unknown_x or key == self.unknown_y:
           raise ValueError(one2one.__breaks_mapping())
        if key in self.x2y:
            value = self.x2y[key]
            del self.x2y[key]
            del self.y2x[value]
        elif key in self.y2x:
            value = self.y2x[key]
            del self.y2x[key]
            del self.x2y[value]
        else:
            raise KeyError(one2one.__key_not_found(key))

    def __iter__(self):
        return self.x2y.__iter__() #Is there another way to call this?

    def items(self):
        return self.x2y.items()

    def __len__(self):
        return len(self.x2y)

    def __str__(self):
        return self.x2y.__str__()

    def x2y_dict(self):
        return deepcopy(self.x2y)

    def y2x_dict(self):
        return deepcopy(self.y2x)
    



