#include <iostream>

#include <utilities.h>
#include <serial.h>
#include <scanTrans.h>
#include <testNvidiaVersion.h>

int main(int argc, char **argv) {

    // Recupero titolo file input
    char    *filename = detectFile(argc, argv[1]);    

    int     m, n, nnz;
    int     *csrRowPtr, *csrColIdx;
    double  *csrVal;

    // Recupero della matrice
    readMatrix(filename, m, n, nnz, csrRowPtr, csrColIdx, csrVal);

    int     *cscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *cscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *cscVal     = (double *)malloc(nnz * sizeof(double));
    
    // Esecuzione dell'algoritmo di trasposizione seriale
    performTransposition(serial, m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
        
    // Esecuzione dell'algoritmo di trasposizione seriale
    // double scanTransTime = performTransposition(scanTrans, m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
    // std::cout << "scanTrans Transposition: " << scanTransTime << " ms\n";


    // TEST NVIDIA

    performTransposition(cuda_sptrans, m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);

    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);
    free(cscRowIdx);
    free(cscColPtr);
    free(cscVal);
}