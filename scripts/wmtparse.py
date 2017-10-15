#External modules
import xml.etree.ElementTree as ET
import itertools

#Internal modules
import ctf_writer

"""
Prepends the given document namespace to all the tag names in a given xmlPath
"""
def addNamespaceGeneric(xmlPath, namespace):
    if not(namespace): return xmlPath

    #Get the individual tag names
    tags = xmlPath.split('/')
    if len(tags) == 0 : return ""
    formattedNamespace = '{' + namespace + '}'
    #Prepend the namespace to each tag
    newPath = formattedNamespace + tags[0]
    for i in range(1, len(tags)):
        newPath += '/' + formattedNamespace + tags[i]
    return newPath

namespace = ""
addNamespace = lambda x: addNamespaceGeneric(x, namespace)

def num_sequences(root):
    count = 0
    docs = root.findall(addNamespace("doc"))

    for doc in docs:
        count += len(list(doc))

    print(count)
    return count

def debug_num_sequences(srcRoot, destRoot):
    srcCount = num_sequences(srcRoot)
    destCount = num_sequences(destRoot)

    if srcCount != destCount:
        print("Error: Unequal number of sequences in source and parallel corpora")


numSentences = 25
sourceLangPath = "../testing/cleaned-newstest2014-deen-src.de.sgm"
destLangPath = "../testing/cleaned-newstest2014-deen-src.en.sgm"
ctfPath = "../data/newstest-2014-deen.ctf"

sourceColumn = "S0"
destColumn = "S1"


#parser = ET.XMLParser(encoding = "utf-8")

srcXML = ET.parse(sourceLangPath) #Load XML data
srcRoot = srcXML.getroot() #Get pointer to root of XML document tree

destXML = ET.parse(destLangPath)
destRoot = destXML.getroot()

#debug_num_sequences(srcRoot, destRoot)


srcDocs = srcRoot.findall(addNamespace("doc"))
destDocs = destRoot.findall(addNamespace("doc"))

ctf = ctf_writer.CTFFile(ctfPath, sourceColumn, destColumn)

for srcDoc, destDoc in itertools.zip_longest(srcDocs, destDocs, fillvalue = "ERROR"):
    srcSegments = srcDoc.findall(addNamespace("seg"))
    destSegments = destDoc.findall(addNamespace("seg"))

    for srcSegment, destSegment in itertools.zip_longest(srcSegments, destSegments, fillvalue = "ERROR"):
        ctf.writeSequence(srcSegment.text, destSegment.text)

ctf.close()
#outFile = open("debug.txt", 'w') #Currently for debugging purposes



#outFile.close()
