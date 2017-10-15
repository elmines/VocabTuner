toConvert = {
    "&": "and"
}

def cleanXML(path):
    index = path.rfind("/")
    outputPath = path[:index + 1] + "cleaned-" + path[index + 1:]

    input = open(path, "r")
    output = open(outputPath, "w")

    nextChar = input.read(1)
    while nextChar:
        if nextChar in toConvert: nextChar = toConvert[nextChar]
        output.write(nextChar)
        #print(nextChar, end = "")
        nextChar = input.read(1)

    input.close()
    output.close()


paths = ["../corpora/newstest2014-deen-src.de.sgm", "../corpora/newstest2014-deen-src.en.sgm"]
for path in paths:
    cleanXML(path) 

"""
test = open("input.txt", "r")
nextChar = test.read(1)
while nextChar:
    print(nextChar)
    print(len(nextChar))
    print("    " + str(nextChar in toConvert))
    nextChar = test.read(1)
"""
