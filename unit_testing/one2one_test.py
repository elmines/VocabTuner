#Import one2one module
import os
import sys
modulesPath = "../scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)
from one2one import one2one


bidict = one2one(unknown_val = "<UNK>")

keys = ["Roderick", "Greg", "Manney", "Fregley", "Chirag", "Rowley"]

i = 0
for key in keys:
    bidict[key] = i
    i += 1

print("Initial dictionary")
print(bidict)
print("Length =", len(bidict))
print()

print("Iterating over keys in domain")
for key in bidict:
    print(key, "<-->", bidict[key], end = "")
    print("\t", bidict[key], "<-->", bidict[bidict[key]])
print()

print("Iterating again with items()")
for key, value in bidict.items():
    print(key, "<-->", value)
print()

print("Searching for two items not in domain or codomain.")
print(bidict["elmines"])
print(bidict[-42])
print()

print("Deleting two mappings from dictionary.")
del bidict[0]
del bidict[keys[1]]
print(bidict.x2y_dict())
print(bidict.y2x_dict())
print()

print("Trying to delete an item that doesn't exist.")
try:
    del bidict[-42]
except KeyError as e:
    print(e.args[0])
print()

print("Modifying two items.")
bidict["Chirag"] = -5 
bidict[5] = "Rabble-rousing Rowley"
print(bidict)
print()

print("Making a new dictionary within nothing set for unknown_val, and searching for a nonexistent key.")
dual_dict = one2one()
try:
    print(dual_dict["elmines"])
except KeyError as e:
    print(e.args[0])