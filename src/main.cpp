#include <iostream>

#include <utilities.h>
#include <serial.h>

int main(int argc, char **argv) {

    // Recupero titolo file input
    char    *filename = detectFile(argc, argv[1]);    

    int     m, n, nnz;
    int     *csrRowPtr, *csrColIdx;
    double    *csrVal;

    // Recupero della matrice
    readMatrix(filename, m, n, nnz, csrRowPtr, csrColIdx, csrVal);

    int     *cscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *cscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double    *cscVal     = (double *)malloc(nnz * sizeof(double));
    
    // Esecuzione dell'algoritmo di trasposizione seriale
    double serialTime = performTransposition(serial, m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
    std::cout << "Serial Transposition: " << serialTime << " ms\n";

    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);
    free(cscRowIdx);
    free(cscColPtr);
    free(cscVal);
}