#include <iostream>
#include <iomanip>

#include <utilities.h>
#include <serial.h>
#include <nvidia.h>
#include <scanTrans.h>

bool checkAllResults(SerialResult serialResult, ScanTransResult scanTransResult);
void checkResults(int m, int *arrayA, int *arrayB);
void checkResults(int m, double *arrayA, double *arrayB);

void nvidiaAlgo1(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
void nvidiaAlgo2(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
void scanTrans(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);


struct ScanTransResult {
    int *cscRowIdx;
    int *cscColPtr;
    double *cscVal;
};

struct SerialResult {
    int *serialCscRowIdx;
    int *serialCscColPtr;
    double *serialCscVal;
} serialResult;

int main(int argc, char **argv) {
    char *filename = detectFile(argc, argv[1]);    
    int m;
    int n;
    int nnz;    
    int *csrRowPtr;
    int *csrColIdx;
    double *csrVal;

    readMatrix(
        filename,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal
    );

    // Serial algorithm
    // serialResult.serialCscRowIdx = (int *)malloc(nnz * sizeof(int));
    // serialResult.serialCscColPtr = (int *)malloc((n + 1) * sizeof(int));
    // serialResult.serialCscVal = (double *)malloc(nnz * sizeof(double));
    float serialTime = performTransposition(
        serial,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal,
        serialCscColPtr,
        serialCscRowIdx,
        serialCscVal
    );
    std::cout << std::endl;

    nvidiaAlgo1(m, n, nnz, csrRowPtr, csrColIdx, csrVal);
    std::cout << std::endl;
    
    nvidiaAlgo2(m, n, nnz, csrRowPtr, csrColIdx, csrVal); 
    std::cout << std::endl;

    scanTransResult = scanTrans(m, n, nnz, *csrRowPtr, *csrColIdx, *csrVal);
    std::cout << std::endl;

    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);

    // free(serialCscRowIdx);
    // free(serialCscColPtr);
    // free(serialCscVal);   
}

void nvidiaAlgo1(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal) {
    int     *nvidiaAlgo1CscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *nvidiaAlgo1CscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *nvidiaAlgo1CscVal     = (double *)malloc(nnz * sizeof(double));
    float nvidiaAlgo1Time = performTransposition(
        nvidia,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal,
        nvidiaAlgo1CscColPtr,
        nvidiaAlgo1CscRowIdx,
        nvidiaAlgo1CscVal
    );
    if(nvidiaAlgo1Time == -1) {
        std::cout << "GPU Sparse Matrix Transpostion ALGO1: memory is too low" << std::endl;
        std::cout << "ALGO1 speedup: -" << std::endl;
    } else {
        std::cout << std::setprecision(1) << "ALGO1 speedup: " << serialTime / nvidiaAlgo1Time << "x" << std::endl;
    }
    // Cleaning device
    cudaDeviceReset();
    free(nvidiaAlgo1CscRowIdx);
    free(nvidiaAlgo1CscColPtr);
    free(nvidiaAlgo1CscVal);
}

void nvidiaAlgo2(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal) {
    int *nvidiaAlgo2CscRowIdx = (int *)malloc(nnz * sizeof(int));
    int *nvidiaAlgo2CscColPtr = (int *)malloc((n + 1) * sizeof(int));
    double *nvidiaAlgo2CscVal = (double *)malloc(nnz * sizeof(double));
    float nvidiaAlgo2Time = performTransposition(
        nvidia2,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal,
        nvidiaAlgo2CscColPtr,
        nvidiaAlgo2CscRowIdx,
        nvidiaAlgo2CscVal
    ); 

    if(nvidiaAlgo2Time == -1) {
        std::cout << "GPU Sparse Matrix Transpostion ALGO2: memory is too low" << std::endl;
        std::cout << "ALGO2 speedup: -" << std::endl;
    } 
    else {
        std::cout << std::setprecision(1) << "ALGO2 speedup: " << serialTime / nvidiaAlgo2Time << "x" << std::endl;
    }
    // Cleaning device
    cudaDeviceReset();
    free(nvidia2CscColPtr);
    free(nvidia2CscRowIdx);
    free(nvidia2CscVal);
}

void scanTrans(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal) {
    ScanTransResult scanTransResult;
    scanTransResult.cscRowIdx = (int *)malloc(nnz * sizeof(int));
    scanTransResult.cscColPtr = (int *)malloc((n + 1) * sizeof(int));
    scanTransResult.cscVal = (double *)malloc(nnz * sizeof(double));
    float scanTransTime = performTransposition(
        scanTrans,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal,
        scanTransCscColPtr,
        scanTransCscRowIdx,
        scanTransCscVal
    ); 
    if(scanTransTime == -1) {
        std::cout << "GPU Sparse Matrix Transpostion ScanTrans: memory is too low" << std::endl;
        std::cout << "ScanTrans wrong: -" << std::endl;
        std::cout << "ScanTrans speedup: -" << std::endl;
    } else {
        bool isWrong = checkAllResults(serialResult, scanTransResult);
        if(wrong)
            std::exit(EXIT_FAILURE);
        else
            std::cout << std::setprecision(2) << "ScanTrans speedup: " << serialTime / scanTransTime << "x" << std::endl;
    }
    // Cleaning device
    cudaDeviceReset();
    // free(scanTransCscRowIdx);
    // free(scanTransCscColPtr);
    // free(scanTransCscVal);
}

bool checkAllResults(SerialResult serialResult, ScanTransResult scanTransResult) {
    bool wrong = false;
    wrong = checkResults(n + 1, serialResult.serialCscColPtr, scanTransResult.cscColPtr);
    if(wrong) return wrong;
    wrong = checkResults(nnz, serialResult.serialCscRowIdx, scanTransResult.cscRowIdx);
    if(wrong) return wrong;
    wrong = checkResults(nnz, serialResult.serialCscVal, scanTransResult.cscVal);
    if(wrong) return wrong;
}

bool checkResultsIsWrong(int m, int *arrayA, int *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            cudaDeviceReset();
            return true;
        }
    }
    return false;
}

bool checkResultsIsWrong(int m, double *arrayA, double *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            cudaDeviceReset();
            return true;
        }
    }
    return false;
}