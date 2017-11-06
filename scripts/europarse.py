import sys

langPair = sys.argv[1].split("-")
sourceLang, destLang = langPair[0], langPair[1]

cleanedPrefix = "cleaned-"
docPrefix = "euroParl-v7."
corpDir = "../corpora"

sourceCorp = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", sourceLang])
destCorp = "".join([corpDir, "/", docPrefix, sourceLang, "-", destLang, ".", destLang])

print(sourceCorp, destCorp)
#sourceCorp = corpDir 
