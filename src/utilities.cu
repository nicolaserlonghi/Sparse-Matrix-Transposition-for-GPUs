#include <utilities.h>
#include <matio.h>

void printArray(int m, int *array) {
    for(int i = 0; i < m; i++) {
        std::cout << array[i] << "\t";
    }
    std::cout << std::endl;
}

char* detectFile(int argc, char* argv) {
    char *filename = NULL;
    if(argc > 1) {
        filename = argv;
    }
    if(filename == NULL) {
        std::cerr << "Error: No file provided!\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "matrix: " << filename << std::endl;
    return filename;
}

void readMatrix(
    char    *filename,
    int     &m,
    int     &n,
    int     &nnz,
    int     *&csrRowPtr,
    int     *&csrColIdx,
    double  *&csrVal
) {
    int retCode = read_mtx_mat(
                                m,
                                n,
                                nnz,
                                csrRowPtr,
                                csrColIdx,
                                csrVal,
                                filename
                            );
    if(retCode != 0) {
        std::cerr << "Failed to read the matrix from " << filename << "!\n";
        std::exit(EXIT_FAILURE);
    }
}

void clearTheBuffers(
    int     n,
    int     nnz,
    int     *cscRowIdx,
    double  *cscVal,
    int     *cscColPtr
) {
    std::fill_n(cscRowIdx, nnz, 0);
    std::fill_n(cscVal, nnz, 0);
    std::fill_n(cscColPtr, n+1, 0);
}

float performTransposition(
    float    (*f)(int, int, int, int*, int*, double*, int*, int*, double*),
    int     m,
    int     n,
    int     nnz,
    int     *csrRowPtr,
    int     *csrColIdx,
    double  *csrVal,
    int     *cscColPtr,
    int     *cscRowIdx,
    double  *cscVal
) {

    // Pulizia dei buffer
    clearTheBuffers(
                    n,
                    nnz,
                    cscRowIdx,
                    cscVal,
                    cscColPtr
                );

    // Chiamata della funzione per la trasposizione
    float result = (*f)(
            m,
            n,
            nnz,
            csrRowPtr,
            csrColIdx,
            csrVal,
            cscColPtr,
            cscRowIdx,
            cscVal
        );
    return result;
}

double getSizeOfNvidiaFreeMemory() {
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    return static_cast<double>(free);
}