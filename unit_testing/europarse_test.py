import os
import sys
modulesPath = "../scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)

import europarse


europarse.parseCorpora("es", "en", numWords = 30, numSequences = 15)
