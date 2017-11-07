import sys

langPair = sys.argv[1].split("-")
sourceLang, destLang = langPair[0], langPair[1]

cleanedPrefix = "cleaned-"
docPrefix = "euroParl-v7."
corpDir = "../corpora"

sourceCorp = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", sourceLang])
destCorp = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", destLang])

numWords = 30000


print(sourceCorp, destCorp)


with open(sourceCorp, "r", encoding = "utf-8") as sourceFile:
    sourceLines = sourceFile.readlines()

with open(destCorp, "r", encoding = "utf-8") as destFile:
    destLines = destFile.readlines()



