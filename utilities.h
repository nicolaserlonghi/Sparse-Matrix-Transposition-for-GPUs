#ifndef _UTILITIES_H
#define _UTILITIES_H

template<typename iT, typename vT>
void print(
    int  m,
    int  n,  
    int  nnz,  
    iT  *cscColPtr,    
    iT  *cscRowIdx,
    vT  *cscVal
) {
    for(int i = 0; i < nnz; i++) {
        std::cout << cscRowIdx[i] << "\t";
    }
    std::cout << std::endl;
    for(int i = 0; i < nnz; i++) {
        std::cout << cscVal[i] << "\t";
    }
    std::cout << std::endl;
}
#endif