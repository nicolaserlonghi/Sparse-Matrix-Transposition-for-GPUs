#include <utilities.h>
#include <matio.h>

void printArray(int  m, double  *array) {
    for(int i = 0; i < m; i++) {
        std::cout << array[i] << "\t";
    }
    std::cout << std::endl;
}

double dtime() {
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime, (struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return (tseconds*1.0e3);
}

char* detectFile(int argc, char* argv) {
    char *filename = NULL;
    if(argc > 1)
    {
        filename = argv;
    }
    if(filename == NULL)
    {
        std::cerr << "Error: No file provided!\n";
    }
    std::cout << "matrix: " << filename << std::endl;
    return filename;
}

void readMatrix(char *filename, int &m, int &n, int &nnz, int *&csrRowPtr, int *&csrColIdx, double *&csrVal) {
    int retCode = read_mtx_mat(m, n, nnz, csrRowPtr, csrColIdx, csrVal, filename);
    if(retCode != 0)
    {
        std::cerr << "Failed to read the matrix from " << filename << "!\n";
    }
}

void clearTheBuffers(int n, int nnz, int *cscRowIdx, double *cscVal, int *cscColPtr) {
    std::fill_n(cscRowIdx, nnz, 0);
    std::fill_n(cscVal, nnz, 0);
    std::fill_n(cscColPtr, n+1, 0);
}

double performTransposition(void (*f)(int, int, int, int*, int*, double*, int*, int*, double*), int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, double *csrVal, int *cscColPtr, int *cscRowIdx, double *cscVal) {
    clearTheBuffers(n, nnz, cscRowIdx, cscVal, cscColPtr);

    double tstart = dtime();

    (*f)(m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);

    double tstop = dtime();

    return tstop - tstart;
}
