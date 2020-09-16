#include <iostream>

#include <utilities.h>
#include <serial.h>
#include <scanTrans.h>
#include <nvidia.h>

void checkResults(int m, int *arrayA, int *arrayB);
void checkResults(int m, double *arrayA, double *arrayB);

int main(int argc, char **argv) {

    // Recupero titolo file input
    char    *filename = detectFile(argc, argv[1]);    
    int     m;
    int     n;
    int     nnz;    
    int     *csrRowPtr;
    int     *csrColIdx;
    double  *csrVal;

    // Recupero della matrice
    readMatrix(
                filename,
                m,
                n,
                nnz,
                csrRowPtr,
                csrColIdx,
                csrVal
            );

    int     *serialCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *serialCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *serialCscVal     = (double *)malloc(nnz * sizeof(double));
    
    // Esecuzione dell'algoritmo di trasposizione seriale
    performTransposition(
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

    int     *nvidiaCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *nvidiaCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *nvidiaCscVal     = (double *)malloc(nnz * sizeof(double));

    // Esecuzione dell'algoritmo di trasposizione versione Nvidia
    performTransposition(
                        nvidia,
                        m,
                        n,
                        nnz,
                        csrRowPtr,
                        csrColIdx,
                        csrVal,
                        nvidiaCscColPtr,
                        nvidiaCscRowIdx,
                        nvidiaCscVal
                    );    

    checkResults(n + 1, serialCscColPtr, nvidiaCscColPtr);
    checkResults(nnz, serialCscRowIdx, nvidiaCscRowIdx);
    checkResults(nnz, serialCscVal, nvidiaCscVal);

    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);
    free(serialCscRowIdx);
    free(serialCscColPtr);
    free(serialCscVal);
    free(nvidiaCscRowIdx);
    free(nvidiaCscColPtr);
    free(nvidiaCscVal);
}

void checkResults(int m, int *arrayA, int *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << arrayA[i]
                      << "\ndevice: " << arrayB[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "\n<> Correct\n";
}

void checkResults(int m, double *arrayA, double *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            std::cerr << "wrong result at: " << i
                      << "\nhost:   " << arrayA[i]
                      << "\ndevice: " << arrayB[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "\n<> Correct\n";
}