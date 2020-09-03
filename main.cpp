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
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Serial version
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int *cscRowIdxA = (int *)malloc(nnzA * sizeof(int));
    int *cscColPtrA = (int *)malloc((n + 1) * sizeof(int));
    valT *cscValA = (valT *)malloc(nnzA * sizeof(valT));
    // clear the buffers
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);

    tstart = dtime();

    serialTransposition<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscColPtrA, cscRowIdxA, cscValA);

    tstop = dtime();
    ttime = tstop - tstart;
    std::cout << "serialVersion(time): " << ttime << " ms\n";

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // clear the buffers
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);

    tstart = dtime();

    sptrans_scanTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscRowIdxA, cscColPtrA, cscValA);

    tstop = dtime();
    ttime = tstop - tstart;
    std::cout << "scanTrans(time): " << ttime << " ms\n";

    // clear the buffers
    std::fill_n(cscRowIdxA, nnzA, 0);
    std::fill_n(cscValA, nnzA, 0);
    std::fill_n(cscColPtrA, n+1, 0);

    tstart = dtime();

    sptrans_mergeTrans<int, valT>(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, cscRowIdxA, cscColPtrA, cscValA);

    tstop = dtime();
    ttime = tstop - tstart;
    std::cout << "mergeTrans(time): " << ttime << " ms\n";

    free(csrRowPtrA); 
    free(csrColIdxA); 
    free(csrValA);
    free(cscRowIdxA);
    free(cscColPtrA);
    free(cscValA);
#ifdef MKL
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(AT);
#endif
}
