#include <iostream>
#include <iomanip>
#include <utilities.h>
#include <serial.h>
#include <nvidia.h>
#include <scanTrans.h>

using namespace std;

// Global variables
int *serialCscRowIdx;
int *serialCscColPtr;
double *serialCscVal;
float serialTime;

// Function prototypes
void serialAlgo(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
void nvidiaAlgo1(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
void nvidiaAlgo2(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
void scanTrans(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal);
bool checkAllResults(
    int *scanTransCscRowIdx, int *scanTransCscColPtr, double *scanTransCscVal,int n, int nnz);
bool checkResultsIsWrong(int m, int *arrayA, int *arrayB);
bool checkResultsIsWrong(int m, double *arrayA, double *arrayB);


int main(int argc, char **argv) {
    char *filename = detectFile(argc, argv[1]);    
    int m;
    int n;
    int nnz;    
    int *csrRowPtr;
    int *csrColIdx;
    double *csrVal;

    readMatrix(filename, m, n, nnz, csrRowPtr, csrColIdx, csrVal);
    
    // Launch of the various algorithms 
    serialAlgo(m, n, nnz, csrRowPtr, csrColIdx, csrVal);
    cout << endl;
    nvidiaAlgo1(m, n, nnz, csrRowPtr, csrColIdx, csrVal);
    cout << endl;
    nvidiaAlgo2(m, n, nnz, csrRowPtr, csrColIdx, csrVal); 
    cout << endl;
    scanTrans(m, n, nnz, csrRowPtr, csrColIdx, csrVal);
    cout << endl;

    // Cleaning
    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);
    free(serialCscRowIdx);
    free(serialCscColPtr);
    free(serialCscVal);
    return 0; 
}

void serialAlgo(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal) {
    serialCscRowIdx = (int *)malloc(nnz * sizeof(int));
    serialCscColPtr = (int *)malloc((n + 1) * sizeof(int));
    serialCscVal = (double *)malloc(nnz * sizeof(double));
    serialTime = performTransposition(
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
        cout << "GPU Sparse Matrix Transpostion ALGO1: memory is too low" << endl;
        cout << "ALGO1 speedup: -" << endl;
    } else {
        cout << setprecision(1) << "ALGO1 speedup: " << serialTime / nvidiaAlgo1Time << "x" << endl;
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
        cout << "GPU Sparse Matrix Transpostion ALGO2: memory is too low" << endl;
        cout << "ALGO2 speedup: -" << endl;
    } 
    else {
        cout << setprecision(1) << "ALGO2 speedup: " << serialTime / nvidiaAlgo2Time << "x" << endl;
    }
    // Cleaning device
    cudaDeviceReset();
    free(nvidiaAlgo2CscColPtr);
    free(nvidiaAlgo2CscRowIdx);
    free(nvidiaAlgo2CscVal);
}

void scanTrans(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal) {
    int *cscRowIdx = (int *)malloc(nnz * sizeof(int));
    int *cscColPtr = (int *)malloc((n + 1) * sizeof(int));
    double *cscVal = (double *)malloc(nnz * sizeof(double));
    float scanTransTime = performTransposition(
        scanTrans,
        m,
        n,
        nnz,
        csrRowPtr,
        csrColIdx,
        csrVal,
        cscColPtr,
        cscRowIdx,
        cscVal
    ); 
    if(scanTransTime == -1) {
        cout << "GPU Sparse Matrix Transpostion ScanTrans: memory is too low" << endl;
        cout << "ScanTrans wrong: -" << endl;
        cout << "ScanTrans speedup: -" << endl;
    } else {
        bool isWrong = checkAllResults(
            cscRowIdx, 
            cscColPtr,
            cscVal,
            n, 
            nnz
        );
        cout << "ScanTrans wrong: " << isWrong << endl;
        if(isWrong)
            exit(EXIT_FAILURE);
        else
            cout << setprecision(1) << "ScanTrans speedup: " << serialTime / scanTransTime << "x" << endl;
    }
    // Cleaning device
    cudaDeviceReset();
    free(cscRowIdx);
    free(cscColPtr);
    free(cscVal);
}

bool checkAllResults(
    int *scanTransCscRowIdx,
    int *scanTransCscColPtr,
    double *scanTransCscVal,
    int n,
    int nnz
) {
    bool isWrong = false;
    isWrong = checkResultsIsWrong(n + 1, serialCscColPtr, scanTransCscColPtr);
    if(isWrong) return isWrong;
    isWrong = checkResultsIsWrong(nnz, serialCscRowIdx, scanTransCscRowIdx);
    if(isWrong) return isWrong;
    isWrong = checkResultsIsWrong(nnz, serialCscVal, scanTransCscVal);
    if(isWrong) return isWrong;
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