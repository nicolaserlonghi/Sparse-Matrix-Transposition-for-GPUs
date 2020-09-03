/*
* (c) 2017 Virginia Polytechnic Institute & State University (Virginia Tech)   
*                                                                              
*   This program is free software: you can redistribute it and/or modify       
*   it under the terms of the GNU Lesser General Public License Version 2.1.                                  
*                                                                              
*   This program is distributed in the hope that it will be useful,            
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              
*   LICENSE in the root of the repository for details.                 
*                                                                              
*/


/* For testing of different SpTrans solutions */
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <sys/time.h> // timing
#include "matio.h"
#include "sptrans.h"
#include "serialTransposition.h"
#include "utilities.h"

#define valT double

void clearTheBuffers(int n, int nnzA, int *cscRowIdxA, valT *cscValA, int *cscColPtrA);

void trySerialVersion(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
);

void try_sptrans_scanTrans(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
);

void try_sptrans_mergeTrans(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
);

int main(int argc, char **argv)
{
    char *filename = NULL;
    if(argc > 1)
    {
        filename = argv[1];
    }
    if(filename == NULL)
    {
        std::cout << "Error: No file provided!\n";
        return EXIT_FAILURE;
    }
    std::cout << "matrix: " << filename << std::endl;

    // input
    // m numero di righe
    // n numero di colonne
    // nnz elementi non nulli
    int m, n, nnzA;
    int *csrRowPtrA;    // Memorizza i puntatori di inizio e fine degli elementi non zero della riga
    int *csrColIdxA;    // Memorizza l'indice degli elementi non zero della colonna
    valT *csrValA;      // Memorizza il valore dell'elemento non nullo
    int retCode = read_mtx_mat(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, filename);
    if(retCode != 0)
    {
        std::cout << "Failed to read the matrix from " << filename << "!\n";
        return EXIT_FAILURE;
    }

    double tstart, tstop, ttime;

    int *cscRowIdxA = (int *)malloc(nnzA * sizeof(int));
    int *cscColPtrA = (int *)malloc((n + 1) * sizeof(int));
    valT *cscValA = (valT *)malloc(nnzA * sizeof(valT));

    
    trySerialVersion(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);
    try_sptrans_scanTrans(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);
    try_sptrans_mergeTrans(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);


    free(csrRowPtrA); 
    free(csrColIdxA); 
    free(csrValA);
    free(cscRowIdxA);
    free(cscColPtrA);
    free(cscValA);
}

void clearTheBuffers(int n, int nnzA, int *cscRowIdxA, valT *cscValA, int *cscColPtrA) {
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);
}

void trySerialVersion(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
) {
    clearTheBuffers(n, nnzA, cscRowIdxA, cscValA, cscColPtrA);

    double tstart = dtime();

    serialTransposition<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);

    double tstop = dtime();
    double ttime = tstop - tstart;
    std::cout << "serialVersion(time): " << ttime << " ms\n";
}

void try_sptrans_scanTrans(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
) {
    clearTheBuffers(n, nnzA, cscRowIdxA, cscValA, cscColPtrA);

    double tstart = dtime();

    sptrans_scanTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);

    double tstop = dtime();
    double ttime = tstop - tstart;
    std::cout << "scanTrans(time): " << ttime << " ms\n";
}

void try_sptrans_mergeTrans(
    int  m,
    int  n,
    int  nnzA,
    int  *csrRowPtrA,
    int  *csrColIdxA,
    valT  *csrValA,    
    int  *cscColPtrA,
    int  *cscRowIdxA,
    valT  *cscValA
) {
    clearTheBuffers(n, nnzA, cscRowIdxA, cscValA, cscColPtrA);

    double tstart = dtime();

    sptrans_scanTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);

    double tstop = dtime();
    double ttime = tstop - tstart;
    std::cout << "mergeTrans(time): " << ttime << " ms\n";
}