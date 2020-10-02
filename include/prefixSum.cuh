#ifndef _PREFIX_SUM_H
#define _PREFIX_SUM_H

#include <utilities.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define ZERO_BANK_CONFLICTS
    #ifdef ZERO_BANK_CONFLICTS
        #define CONFLICT_FREE_OFFSET(n) ( ((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)) )
    #else
        #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

// Extra shared memory space because of the padding
#define EXTRA (CONFLICT_FREE_OFFSET((NUM_THREADS * 2 - 1))

#define CUDA_ERROR( err, msg ) { \
    if (err != cudaSuccess) {\
        printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
        exit( EXIT_FAILURE );\
    }\
}

// Function prototypes
void prefixSum(int *d_input, int *d_cscColPtr, int numElements);
int manageMemoryForPrefixSum(int numElements);

#endif