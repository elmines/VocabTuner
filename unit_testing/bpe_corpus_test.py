import os
import sys


modulesPath = "../scripts"
modulesPath = os.path.abspath(os.path.join(modulesPath))
if modulesPath not in sys.path: sys.path.append(modulesPath)

from bpe_corpus import BPECorpus

sourcePath = "../corpora/europarl-v7.es-en.es"
destPath = "../corpora/europarl-v7.es-en.en"

with open(sourcePath, "r") as sourceFile:
    sourceLines = sourceFile.readlines()
with open(destPath, "r") as destFile:
    destLines = destFile.readlines()

corp = BPECorpus(sourceLines, destLines)
