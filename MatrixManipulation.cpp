#include <string>
#include <sstream>
#include <vector>
#include"MatrixManipulation.h"
#include<iostream>

using namespace std;

    // User defined exceptions
class invalid_Matrices_input : public exception
{
public:
    const char * what() const throw()
    {
        return "ERROR: cannot multiply the two Matrices!!!";
    }
};

    // Checks if the size of the two matrices are compatible for dot product
template<typename T >
bool MatrixManipulation<T>::check_dot_product(Matrix<T> a, Matrix<T> b)
{
    if (a[0].size() != b.size())
    {
        return false;
    }
    return true;
}

    // Checks if the size of the two matrices are compatible for sum
template<typename T >
bool MatrixManipulation<T>::check_sum(Matrix<T> a, Matrix<T> b)
{
    if (a.size() == b.size() && a[0].size() == b[0].size())
    {
        return true;
    }
    return false;
}

    // The dot product
template<typename T>
Matrix<T> MatrixManipulation<T>::dot_product(Matrix<T> a, Matrix<T> b)
{
    // Check if the sizes of matrices are compatible
    if (!check_dot_product(a, b))
    {
        invalid_Matrices_input err;
        throw err;
    }
    Matrix<T> c;
    for (unsigned int i = 0; i < a.size(); i++)
    {
        c.push_back({});
        for (unsigned int j = 0; j < b[0].size(); j++)
        {
            T temp =  0;
            for (unsigned int k = 0; k < a[0].size(); k++)
            {
                temp += a[i][k] * b[k][j];  // rows of first matrix * columns of second matrix
            }
            c[i].push_back(temp);
        }
    }
    return c;
}

    //Transpose of a matrix input
template<typename T>
Matrix<T> MatrixManipulation<T>::tranpose(Matrix<T> m)
{
    Matrix<T> maxtr;
    for (unsigned int col = 0; col < m[0].size(); col++)
    {
        maxtr.push_back({});
        for (unsigned int row = 0; row < m.size(); row++)
        {
            maxtr[col].push_back(m[row][col]);
        }
    }
    return maxtr;
}

    // Tranpose a vector into a matrix. The new Matrix have column = 1, row = vector's size
template<typename T>
Matrix<T> MatrixManipulation<T>::tranpose(Vect<T> v)
{
    Matrix<T> maxtr;
    for (unsigned int row = 0; row < v.size(); row++)
    {
        maxtr.push_back({});            // one column
        maxtr[row].push_back(v[row]);   // Push each element into each row
    }
    return maxtr;
}

template<typename T>
Matrix<T> MatrixManipulation<T>::sum(Matrix<T> a, Matrix<T> b)
{
    // check if two Matrix have the same size
    if (!check_sum(a, b))
    {
        invalid_Matrices_input err;
        throw err;
    }
    Matrix<T> m;
    for (unsigned int i = 0; i < a.size(); i++)
    {
        m.push_back({});
        for (unsigned int j = 0; j < a[i].size(); j++)
        {
            m[i].push_back(a[i][j] + b[i][j]);
        }
    }
    return m;
}

    // Takes a vector and then transfer it into a matrix. New matrix: row = 1, column = vector's size
template<typename T>
Matrix<T> MatrixManipulation<T>::transf(Vect<T> v)
{
    Matrix<T> m;
    m.push_back(v);
    return m;
}


    //Overloading operator * for the dot product
template<typename T>
Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b)
{
    Matrix<T> result = MatrixManipulation<T>().dot_product(a,b);
    return result;
}

    //Overloading operator + for sum of 2 matrices
template<typename T>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b)
{
    Matrix<T> result = MatrixManipulation<T>().sum(a,b);
    return result;
}
