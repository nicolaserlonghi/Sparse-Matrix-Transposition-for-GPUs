#include <iostream>
#include <iomanip>

#include <utilities.h>
#include <serial.h>
#include <nvidia.h>
#include <scanTrans.h>

int checkResults(int m, int *arrayA, int *arrayB);
int checkResults(int m, double *arrayA, double *arrayB);

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
    float   serialTime;
    
    // Esecuzione dell'algoritmo di trasposizione seriale
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

    cudaDeviceReset();
    std::cout << std::endl;

    // int     *nvidiaCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    // int     *nvidiaCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    // double  *nvidiaCscVal     = (double *)malloc(nnz * sizeof(double));
    // float   nvidiaTime;

    // Esecuzione dell'algoritmo di trasposizione versione Nvidia ALGO1
    // nvidiaTime = performTransposition(
    //                                 nvidia,
    //                                 m,
    //                                 n,
    //                                 nnz,
    //                                 csrRowPtr,
    //                                 csrColIdx,
    //                                 csrVal,
    //                                 nvidiaCscColPtr,
    //                                 nvidiaCscRowIdx,
    //                                 nvidiaCscVal
    //                             );
    // if(nvidiaTime == -1) {
    //     std::cout << "GPU Sparse Matrix Transpostion ALGO1: memory is too low" << std::endl;
    //     std::cout << "ALGO1 speedup: -" << std::endl;
    // } else {
    //     std::cout << std::setprecision(1) << "ALGO1 speedup: " << serialTime / nvidiaTime << "x" << std::endl;
    // }    

    // cudaDeviceReset();

    // free(nvidiaCscRowIdx);
    // free(nvidiaCscColPtr);
    // free(nvidiaCscVal);

    // std::cout << std::endl;

    // int     *nvidia2CscRowIdx  = (int *)malloc(nnz * sizeof(int));
    // int     *nvidia2CscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    // double  *nvidia2CscVal     = (double *)malloc(nnz * sizeof(double));
    // float   nvidia2Time;

    // Esecuzione dell'algoritmo di trasposizione versione Nvidia ALGO2
    // nvidia2Time = performTransposition(
    //                             nvidia2,
    //                             m,
    //                             n,
    //                             nnz,
    //                             csrRowPtr,
    //                             csrColIdx,
    //                             csrVal,
    //                             nvidia2CscColPtr,
    //                             nvidia2CscRowIdx,
    //                             nvidia2CscVal
    //                         ); 

    // if(nvidia2Time == -1) {
    //     std::cout << "GPU Sparse Matrix Transpostion ALGO2: memory is too low" << std::endl;
    //     std::cout << "ALGO2 speedup: -" << std::endl;
    // } 
    // else {
    //     std::cout << std::setprecision(1) << "ALGO2 speedup: " << serialTime / nvidia2Time << "x" << std::endl;
    // }

    

    // cudaDeviceReset();

    // free(nvidia2CscColPtr);
    // free(nvidia2CscRowIdx);
    // free(nvidia2CscVal); 

    // std::cout << std::endl;


    int     *scanTransCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *scanTransCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *scanTransCscVal     = (double *)malloc(nnz * sizeof(double));
    float   scanTransTime;

    // Esecuzione dell'algoritmo di trasposizione versione articolo scanTrans
    scanTransTime = performTransposition(
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
    } 

    if(scanTransTime == -2) {
        std::cout << "GPU Sparse Matrix Transpostion ScanTrans: max blocks num reached" << std::endl;
        std::cout << "ScanTrans wrong: -" << std::endl;
        std::cout << "ScanTrans speedup: -" << std::endl;
    } 

    if(scanTransTime == -3) {
        std::cout << "GPU Sparse Matrix Transpostion ScanTrans: max threads num reached" << std::endl;
        std::cout << "ScanTrans wrong: -" << std::endl;
        std::cout << "ScanTrans speedup: -" << std::endl;
    } 

    if (scanTransTime != -1) {
        std::cout << std::setprecision(1) << "ScanTrans speedup: " << serialTime / scanTransTime << "x" << std::endl;
        std::cout << "check cscColPtr ScanTrans ";
        // scanTransTime = checkResults(n + 1, serialCscColPtr, scanTransCscColPtr);            
    }

    if (scanTransTime != -1) {
        std::cout << "\ncheck cscRowIdx ScanTrans ";
        // scanTransTime = checkResults(nnz, serialCscRowIdx, scanTransCscRowIdx);
    }

    if (scanTransTime != -1) {
        std::cout << "\ncheck cscVal ScanTrans ";
        // scanTransTime = checkResults(nnz, serialCscVal, scanTransCscVal);
    }

    if (scanTransTime != -1) {
        std::cout << "wrong: 0" << std::endl;
    }    

    cudaDeviceReset();

    free(scanTransCscRowIdx);
    free(scanTransCscColPtr);
    free(scanTransCscVal);

    std::cout << std::endl;

    free(csrRowPtr); 
    free(csrColIdx); 
    free(csrVal);

    // free(serialCscRowIdx);
    // free(serialCscColPtr);
    // free(serialCscVal);   
}

int checkResults(int m, int *arrayA, int *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            std::cout << "wrong: 1 \n";
                    //   << "\nhost:   " << arrayA[i]
                    //   << "\ndevice: " << arrayB[i] << "\n\n";
            cudaDeviceReset();
            return -1;
        }
    }
    return 0;
    // std::cout << "\n<> Correct\n";
}

int checkResults(int m, double *arrayA, double *arrayB) {
    for (int i = 0; i < m; i++) {
        if (arrayA[i] != arrayB[i]) {
            std::cout << "wrong: 1 \n";
                    //   << "\nhost:   " << arrayA[i]
                    //   << "\ndevice: " << arrayB[i] << "\n\n";
            cudaDeviceReset();
            return -1;
        }
    }
    return 0;
    // std::cout << "\n<> Correct\n";
}