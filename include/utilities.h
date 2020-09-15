#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <iostream>

/**
 * @brief: stampa il contenuto di un array in riga
 * @param: m: la dimensione dell'array
 * @param: array: il puntatore all'array da stampare
 */
void printArray(int m, double *array);


/**
 * @brief: identifica il nome del file da aprire
 * @param: argc: il numero di parametri passati in input al programma
 * @param: argv: un puntatore al nome del file se presente
 * @return: il nome del file
 */
char* detectFile(int argc, char *argv);

/**
 * @brief: chiama un funzione per la lettura della matrice da file
 * @param: filename: il nome del file contenente la matrice
 * @param: m: il numero di righe della matrice
 * @param: n: il numero di colonne della matrice
 * @param: nnz: il numero di elementi diversi da zero nella matrice
 * @param: csrRowPtr: array di dimensione m + 1 contenente il numero accumulativo di elementi diversi da zero in ogni riga
 * @param: csrColIdx: array di dimensione nnz contenente l'indice di colonna di ogni elemento diverso da zero
 * @param: csrVal: array di dimensione nnz contenente gli elementi diversi da zero
 */
void readMatrix(
    char    *filename,
    int     &m,
    int     &n,
    int     &nnz,
    int     *&csrRowPtr,
    int     *&csrColIdx,
    double  *&csrVal
);

/**
 * @brief: azzera gli array della matrice trasposta
 * @param: n: il numero di colonne della matrice originale
 * @param: nnz: il numero di elementi diversi da zero
 * @param: cscRowIdx: array di dimensione nnz contenente l'indice di riga di ogni elemento diverso da zero
 * @param: cscVal: array di dimensione nnz contenente gli elementi diversi da zero
 * @param: cscColPtr: array di dimensione n + 1 contenente il numero accumulativo di elementi diversi da zero in ogni colonna
 */
void clearTheBuffers(
    int     n,
    int     nnz,
    int     *cscRowIdx,
    double  *cscVal,
    int     *cscColPtr
);

/**
 * @brief: calcola la trasposta della matrice originale utilizzando un funzione specifica
 * @param: (*f)...: puntatore alla funzione che verrà usata per la trasposizione
 * @param: m: il numero di righe della matrice originale
 * @param: n: il numero di colonne della matrice originale
 * @param: nnz: il numero di elementi diversi da zero
 * @param: csrRowPtr: array di dimensione m + 1 contenente il numero accumulativo di elementi diversi da zero in ogni riga della matrice originale
 * @param: csrColIdx: array di dimensione nnz contenente l'indice di riga di ogni elemento diverso da zero della matrice originale
 * @param: csrVal: array di dimensione nnz contenente gli elementi diversi da zero della matrice originale
 * @param: cscColPtr: array di dimensione n + 1 contenente il numero accumulativo di elementi diversi da zero in ogni colonna della matrice trasposta
 * @param: cscRowIdx: array di dimensione nnz contenente l'indice di riga di ogni elemento diverso da zero della matrice trasposta
 * @param: cscVal: array di dimensione nnz contenente gli elementi diversi da zero della matrice trasposta
 */
void performTransposition(
    void    (*f)(int, int, int, int*, int*, double*, int*, int*, double*),
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal
);

#endif