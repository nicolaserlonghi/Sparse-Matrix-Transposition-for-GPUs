#ifndef _UTILITIES_H
#define _UTILITIES_H

template<typename iT>
void printArray(
    int  m,
    iT  *array
) {
    for(int i = 0; i < m; i++) {
        std::cout << array[i] << "\t";
    }
    std::cout << std::endl;
}

double dtime()  // milliseconds
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime, (struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return (tseconds*1.0e3);
}
#endif