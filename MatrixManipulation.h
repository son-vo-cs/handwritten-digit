#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED

#include <vector>
#include <string>

    //Alias
template<typename T>
using Matrix = std::vector<std::vector<T>>;

template<typename T>
using Vect = std::vector<T>;

template<typename T>
class MatrixManipulation
{
    //Overloading operator * for the dot_product of 2 matrices
template<typename T1>
friend Matrix<T1> operator*(const Matrix<T1>& a, const Matrix<T1>& b);

    //Overloading operator + for the sum of 2 matrices
template<typename T1>
friend Matrix<T1> operator+(const Matrix<T1>& a, const Matrix<T1>& b);

    //public static functions
public:
        // Checks if the size of the two matrices are compatible for dot product
    static bool check_dot_product(Matrix<T> a, Matrix<T> b);

        // Checks if the size of the two matrices are compatible for sum
    static bool check_sum(Matrix<T> a, Matrix<T> b);

        // Takes two matrices as inputs and compute the dot product
    static Matrix<T> dot_product(Matrix<T> a, Matrix<T> b);

        // Takes a matrix and then return the transpose of that matrix
    static Matrix<T> tranpose(Matrix<T> m);

        // Takes two matrices as inputs and compute the sum
    static Matrix<T> sum(Matrix<T> a, Matrix<T> b);

        // Takes a vector and then return a matrix with row = vector's size and column = 1
    static Matrix<T> tranpose(Vect<T> v);

        // Takes a vector and then turn it into a matrix with only 1 row and column = vector's size
    static Matrix<T> transf(Vect<T> v);



};



#endif // HELPERS_H_INCLUDED
