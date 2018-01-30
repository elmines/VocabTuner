import sys



def readScore(fp):
    scoreLabel = "NIST score = "
    maxRawScore = 100


    line = fp.readline()
    while line and (not line.startswith(scoreLabel)):
        line = fp.readline()
    
    startIndex = line.index(scoreLabel)
    scoreIndex = startIndex + len(scoreLabel)
    scoreString = line[scoreIndex : ].split(" ")[0]

    return float(scoreString)

def convertScore(score):
    maxRawScore = 100
    return maxRawScore - score

if __name__ == "__main__":
    if len(sys.argv) > 1:
       with open(sys.argv[1], "r") as fp:
           score = convertScore(readScore(fp))
    else:
       fp = sys.stdin 
       score = convertScore(readScore(fp))
    print(score) 
