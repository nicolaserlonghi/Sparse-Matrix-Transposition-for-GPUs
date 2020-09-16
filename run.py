import os
import sys
import string
import commands

def excuteCmdAndPrintOutput(cmd):
    result = commands.getoutput(cmd)
    print result
    return result

def printCommentWithHeader(text):
    print("\n#########################################################################################\n")
    print("#\t\t\t\t\t\t\t\t\t\t\t#\n")
    print("\t\t\t" + text + "\n")
    print("#\t\t\t\t\t\t\t\t\t\t\t#\n")
    print("#########################################################################################\n")

def cleanCompile():
    printCommentWithHeader("Rimozione del vecchio eseguibile")
    excuteCmdAndPrintOutput("rm -rf ./bin")

    printCommentWithHeader("Compilazione del progetto e generazione dell'eseguibile")
    excuteCmdAndPrintOutput("make")

    printCommentWithHeader("Installazione dell'eseguibile nella cartella bin")
    excuteCmdAndPrintOutput("make install")

    printCommentWithHeader("Rimozione dei file .o creati")
    excuteCmdAndPrintOutput("make clean")

    printCommentWithHeader("Esecuzione del programma")

def getTextAfterKey(text, key):
    index = text.find(key) + len(key)
    tmp = text[index : -1]
    tmp = tmp.split("\n")[0]
    result = tmp.strip()
    return result

def parseResultAndSaveToFile(resultFile, result, matrixName):
    matrixType = getTextAfterKey(result, "type =")
    m = getTextAfterKey(result, "m: ")
    n = getTextAfterKey(result, "n: ")
    nnz = getTextAfterKey(result, "nnz: ")
    serialTime = getTextAfterKey(result, "Serial Sparse Matrix Transpostion:")
    nvidiaTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion:")
    fileLine = matrixName + "; " + matrixType + "; " +  \
                    m + "; " + n + "; " + nnz + "; " + \
                    serialTime + "; " + nvidiaTime + "\n"
    resultFile.write(fileLine)

def startTest(testPath):
    # save result file
    resultFile = open("results.csv","w")
    resultFile.write("matrix; type; m; n; nnz; serial; nvidia\n")
    for file in os.listdir(testPath):
        if file.endswith(".mtx"):
            filePath = os.path.join(testPath, file)
            cmd = "./bin/sparse_matrix_transpose " + filePath
            result = excuteCmdAndPrintOutput(cmd)
            print("\n#########################################################################################\n")
        parseResultAndSaveToFile(resultFile, result, file)      
    resultFile.write("\n")
    resultFile.close()


# matrix data path
testPath = "testFiles/"

cleanCompile()
startTest(testPath)


printCommentWithHeader("Terminato")