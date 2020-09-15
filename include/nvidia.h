#ifndef _NVIDIA_H
#define _NVIDIA_H

#include <cusparse_v2.h>

/**
 * @brief: esegue la trasposizione della matrice secondo la soluzione di Nvidia
 * @param: m: il numero di righe della matrice
 * @param: n: il numero di colonne della matrice
 * @param: nnz: il numero di elementi diversi da zero nella matrice
 * @param: csrRowPtr: array di dimensione m + 1 contenente il numero accumulativo di elementi diversi da zero in ogni riga della matrice originale
 * @param: csrColIdx: array di dimensione nnz contenente l'indice di riga di ogni elemento diverso da zero della matrice originale
 * @param: csrVal: array di dimensione nnz contenente gli elementi diversi da zero della matrice originale
 * @param: cscColPtr: array di dimensione n + 1 contenente il numero accumulativo di elementi diversi da zero in ogni colonna della matrice trasposta
 * @param: cscRowIdx: array di dimensione nnz contenente l'indice di riga di ogni elemento diverso da zero della matrice trasposta
 * @param: cscVal: array di dimensione nnz contenente gli elementi diversi da zero della matrice trasposta
 */
void nvidia(
    int         m,
    int         n,
    int         nnz,
    int        *csrRowPtr,
    int        *csrColIdx,
    double     *csrVal,
    int        *cscColPtr,
    int        *cscRowIdx,
    double     *cscVal
);

#endif