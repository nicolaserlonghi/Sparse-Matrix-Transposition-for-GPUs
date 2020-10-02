#include <iostream>
#include <iomanip>

#include <utilities.h>
#include <serial.h>
#include <nvidia.h>
#include <scanTransCooperative.h>
#include <scanTransRowRow.h>
#include <scanTrans.h>
#include <scanSpeed.h>

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

    // int device;
    // cudaGetDevice(&device);

    // struct cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, device);

    // if (props.cooperativeLaunch == 1) {

    //     int     *scanTransCooperativeCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    //     int     *scanTransCooperativeCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    //     double  *scanTransCooperativeCscVal     = (double *)malloc(nnz * sizeof(double));
    //     float   scanTransCooperativeTime;

    //     // Esecuzione dell'algoritmo di trasposizione versione articolo ScanTrans Cooperative
    //     scanTransCooperativeTime = performTransposition(
    //                                 scanTransCooperative,
    //                                 m,
    //                                 n,
    //                                 nnz,
    //                                 csrRowPtr,
    //                                 csrColIdx,
    //                                 csrVal,
    //                                 scanTransCooperativeCscColPtr,
    //                                 scanTransCooperativeCscRowIdx,
    //                                 scanTransCooperativeCscVal
    //                             ); 

    //     if(scanTransCooperativeTime == -1) {            
    //         std::cout << "GPU Sparse Matrix Transpostion ScanTrans Cooperative: memory is too low" << std::endl;
    //         std::cout << "ScanTrans Cooperative wrong: -" << std::endl;
    //         std::cout << "ScanTrans Cooperative speedup: -" << std::endl;
    //     } 

    //     if (scanTransCooperativeTime != -1) {
    //         std::cout << std::setprecision(1) << "ScanTrans Cooperative speedup: " << serialTime / scanTransCooperativeTime << "x" << std::endl;
    //         std::cout << "check cscColPtr ScanTrans Cooperative ";
    //         scanTransCooperativeTime = checkResults(n + 1, serialCscColPtr, scanTransCooperativeCscColPtr);            
    //     }

    //     if (scanTransCooperativeTime != -1) {
    //         std::cout << "\ncheck cscRowIdx ScanTrans Cooperative ";
    //         scanTransCooperativeTime = checkResults(nnz, serialCscRowIdx, scanTransCooperativeCscRowIdx);
    //     }

    //     if (scanTransCooperativeTime != -1) {
    //         std::cout << "\ncheck cscVal ScanTrans Cooperative ";
    //         scanTransCooperativeTime = checkResults(nnz, serialCscVal, scanTransCooperativeCscVal);
    //     }

    //     if (scanTransCooperativeTime != -1) {
    //         std::cout << "wrong: 0" << std::endl;
    //     }        

    //     cudaDeviceReset();

    //     free(scanTransCooperativeCscRowIdx);
    //     free(scanTransCooperativeCscColPtr);
    //     free(scanTransCooperativeCscVal);        

    //     std::cout << std::endl;
    // }    
    // else {
        // std::cout << "GPU Sparse Matrix Transpostion ScanTrans Cooperative: - ms" << std::endl;
        // std::cout << "ScanTrans Cooperative wrong: -" << std::endl;
        // std::cout << "ScanTrans Cooperative speedup: -" << std::endl;
        // std::cout << std::endl;
    // }

    // int     *scanTransRowRowCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    // int     *scanTransRowRowCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    // double  *scanTransRowRowCscVal     = (double *)malloc(nnz * sizeof(double));
    // float   scanTransRowRowTime;

    // Esecuzione dell'algoritmo di trasposizione versione articolo ScanTrans Row-Row
    // scanTransRowRowTime = performTransposition(
    //                                         scanTransRowRow,
    //                                         m,
    //                                         n,
    //                                         nnz,
    //                                         csrRowPtr,
    //                                         csrColIdx,
    //                                         csrVal,
    //                                         scanTransRowRowCscColPtr,
    //                                         scanTransRowRowCscRowIdx,
    //                                         scanTransRowRowCscVal
    // ); 

    // if(scanTransRowRowTime == -1) {
    //     std::cout << "GPU Sparse Matrix Transpostion ScanTrans Row-Row: memory is too low" << std::endl;
    //     std::cout << "ScanTrans Row-Row wrong: -" << std::endl;
    //     std::cout << "ScanTrans Row-Row speedup: -" << std::endl;
    // } 

    // if (scanTransRowRowTime != -1) {
    //     std::cout << std::setprecision(1) << "ScanTrans Row-Row speedup: " << serialTime / scanTransRowRowTime << "x" << std::endl;
    //     std::cout << "check cscColPtr ScanTrans Row-Row ";
    //     scanTransRowRowTime = checkResults(n + 1, serialCscColPtr, scanTransRowRowCscColPtr);            
    // }

    // if (scanTransRowRowTime != -1) {
    //     std::cout << "\ncheck cscRowIdx ScanTrans Row-Row ";
    //     scanTransRowRowTime = checkResults(nnz, serialCscRowIdx, scanTransRowRowCscRowIdx);
    // }

    // if (scanTransRowRowTime != -1) {
    //     std::cout << "\ncheck cscVal ScanTrans Row-Row ";
    //     scanTransRowRowTime = checkResults(nnz, serialCscVal, scanTransRowRowCscVal);
    // }

    // if (scanTransRowRowTime != -1) {
    //     std::cout << "wrong: 0" << std::endl;
    // }    

    // cudaDeviceReset();

    // free(scanTransRowRowCscRowIdx);
    // free(scanTransRowRowCscColPtr);
    // free(scanTransRowRowCscVal);

    // std::cout << std::endl;

    // int     *scanTransCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    // int     *scanTransCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    // double  *scanTransCscVal     = (double *)malloc(nnz * sizeof(double));
    // float   scanTransTime;

    // Esecuzione dell'algoritmo di trasposizione versione articolo ScanTrans
    // scanTransTime = performTransposition(
    //                                     scanTrans,
    //                                     m,
    //                                     n,
    //                                     nnz,
    //                                     csrRowPtr,
    //                                     csrColIdx,
    //                                     csrVal,
    //                                     scanTransCscColPtr,
    //                                     scanTransCscRowIdx,
    //                                     scanTransCscVal
    // ); 

    // if(scanTransTime == -1) {
    //     std::cout << "GPU Sparse Matrix Transpostion ScanTrans: memory is too low" << std::endl;
    //     std::cout << "ScanTrans wrong: -" << std::endl;
    //     std::cout << "ScanTrans speedup: -" << std::endl;
    // } 
    // if(scanTransTime == -2) {
    //     std::cout << "GPU Sparse Matrix Transpostion ScanTrans: max blocks num reached" << std::endl;
    //     std::cout << "ScanTrans wrong: -" << std::endl;
    //     std::cout << "ScanTrans speedup: -" << std::endl;
    // } 
    // if(scanTransTime == -3) {
    //     std::cout << "GPU Sparse Matrix Transpostion ScanTrans: max threads num reached" << std::endl;
    //     std::cout << "ScanTrans wrong: -" << std::endl;
    //     std::cout << "ScanTrans speedup: -" << std::endl;
    // } 

    // if (scanTransTime != -1) {
    //     std::cout << std::setprecision(1) << "ScanTrans speedup: " << serialTime / scanTransTime << "x" << std::endl;
    //     std::cout << "check cscColPtr ScanTrans ";
    //     // scanTransTime = checkResults(n + 1, serialCscColPtr, scanTransCscColPtr);            
    // }

    // if (scanTransTime != -1) {
    //     std::cout << "\ncheck cscRowIdx ScanTrans ";
    //     // scanTransTime = checkResults(nnz, serialCscRowIdx, scanTransCscRowIdx);
    // }

    // if (scanTransTime != -1) {
    //     std::cout << "\ncheck cscVal ScanTrans ";
    //     // scanTransTime = checkResults(nnz, serialCscVal, scanTransCscVal);
    // }

    // if (scanTransTime != -1) {
    //     std::cout << "wrong: 0" << std::endl;
    // }    

    // cudaDeviceReset();

    // free(scanTransCscRowIdx);
    // free(scanTransCscColPtr);
    // free(scanTransCscVal);

    // std::cout << std::endl;

    int     *scanSpeedCscRowIdx  = (int *)malloc(nnz * sizeof(int));
    int     *scanSpeedCscColPtr  = (int *)malloc((n + 1) * sizeof(int));
    double  *scanSpeedCscVal     = (double *)malloc(nnz * sizeof(double));
    float   scanSpeedTime;

    // Esecuzione dell'algoritmo di trasposizione versione articolo scanSpeed
    scanSpeedTime = performTransposition(
                                        scanSpeed,
                                        m,
                                        n,
                                        nnz,
                                        csrRowPtr,
                                        csrColIdx,
                                        csrVal,
                                        scanSpeedCscColPtr,
                                        scanSpeedCscRowIdx,
                                        scanSpeedCscVal
    ); 

    if(scanSpeedTime == -1) {
        std::cout << "GPU Sparse Matrix Transpostion ScanSpeed: memory is too low" << std::endl;
        std::cout << "ScanSpeed wrong: -" << std::endl;
        std::cout << "ScanSpeed speedup: -" << std::endl;
    } 

    if(scanSpeedTime == -2) {
        std::cout << "GPU Sparse Matrix Transpostion ScanSpeed: max blocks num reached" << std::endl;
        std::cout << "ScanSpeed wrong: -" << std::endl;
        std::cout << "ScanSpeed speedup: -" << std::endl;
    } 

    if(scanSpeedTime == -3) {
        std::cout << "GPU Sparse Matrix Transpostion ScanSpeed: max threads num reached" << std::endl;
        std::cout << "ScanSpeed wrong: -" << std::endl;
        std::cout << "ScanSpeed speedup: -" << std::endl;
    } 

    if (scanSpeedTime != -1) {
        std::cout << std::setprecision(1) << "ScanSpeed speedup: " << serialTime / scanSpeedTime << "x" << std::endl;
        std::cout << "check cscColPtr ScanSpeed ";
        // scanSpeedTime = checkResults(n + 1, serialCscColPtr, scanSpeedCscColPtr);            
    }

    if (scanSpeedTime != -1) {
        std::cout << "\ncheck cscRowIdx ScanSpeed ";
        // scanSpeedTime = checkResults(nnz, serialCscRowIdx, scanSpeedCscRowIdx);
    }

    if (scanSpeedTime != -1) {
        std::cout << "\ncheck cscVal ScanSpeed ";
        // scanSpeedTime = checkResults(nnz, serialCscVal, scanSpeedCscVal);
    }

    if (scanSpeedTime != -1) {
        std::cout << "wrong: 0" << std::endl;
    }    

    cudaDeviceReset();

    free(scanSpeedCscRowIdx);
    free(scanSpeedCscColPtr);
    free(scanSpeedCscVal);

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