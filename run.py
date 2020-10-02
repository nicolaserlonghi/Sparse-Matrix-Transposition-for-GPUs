import os
import sys
import subprocess
import ntpath

def excuteCmdAndPrintOutput(cmd):
    result = subprocess.getoutput(cmd)
    print(result + "\n")
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
    # nvidiaTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ALGO1:")
    # nvidiaSpeedup = getTextAfterKey(result, "ALGO1 speedup:")
    # nvidiaTimeAlgo2 = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ALGO2:")
    # nvidia2Speedup = getTextAfterKey(result, "ALGO2 speedup:")
    # scanTransCooperativeTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ScanTrans Cooperative:")
    # scanTransCooperativeSpeedup = getTextAfterKey(result, "ScanTrans Cooperative speedup:")
    # wrongCooperativeResult = getTextAfterKey(result, "ScanTrans Cooperative wrong: ")
    # scanTransRowRowTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ScanTrans Row-Row:")
    # scanTransRowRowSpeedup = getTextAfterKey(result, "ScanTrans Row-Row speedup:")
    # wrongRowRowResult = getTextAfterKey(result, "ScanTrans Row-Row wrong: ")
    scanTransTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ScanTrans:")
    scanTransSpeedup = getTextAfterKey(result, "ScanTrans speedup:")
    wrongResult = getTextAfterKey(result, "ScanTrans wrong: ")
    scanSpeedTime = getTextAfterKey(result, "GPU Sparse Matrix Transpostion ScanSpeed:")
    scanSpeedSpeedup = getTextAfterKey(result, "ScanSpeed speedup:")
    wrongSpeedResult = getTextAfterKey(result, "ScanSpeed wrong: ")
    fileLine = matrixName + "; " + matrixType + "; " +  \
                    m + "; " + n + "; " + nnz + "; " + \
                    scanTransTime + "; " + scanTransSpeedup + "; " + wrongResult + "; " + \
                    scanSpeedTime + "; " + scanSpeedSpeedup + "; " + wrongSpeedResult + "\n"
    resultFile.write(fileLine)

def executeCommandOnFile(file, path=''):
    if file.endswith(".mtx"):
        filePath = os.path.join(path, file)
        cmd = "./bin/sparse_matrix_transpose " + filePath
        result = excuteCmdAndPrintOutput(cmd)
        print("\n#########################################################################################\n")
        return result
    else:
        print("ERROR: " + str(file) + " is not a matrix.\n")


def startTest(paths):
    # save result file
    resultFile = open("results.csv","w")
    resultFile.write("matrix; type; m; n; nnz; scanTrans; speedup; wrong; scanSpeed; speedup; wrong\n")
    for path in paths:
        if(os.path.isfile(path)):
            file = ntpath.basename(path)
            result = executeCommandOnFile(path)
            parseResultAndSaveToFile(resultFile, result, file)
        elif os.path.isdir(path): 
            for file in os.listdir(path):
                result = executeCommandOnFile(file, path)
                parseResultAndSaveToFile(resultFile, result, file)
        else:
            print("You have to pass a path of dir or file \n")
    resultFile.write("\n")
    resultFile.close()


# matrix data path
testPath = "testFiles/"

cleanCompile()

args = sys.argv[1:]
if(len(args) == 0):
    startTest([testPath])
else:
    startTest(args)


printCommentWithHeader("Terminato")