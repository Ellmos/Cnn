#include <iostream>

#include "matrix/matrix.hpp"

using namespace std;


int main(void)
{
    Matrix<int> input = Matrix<int>(3, 3);
    input(0, 0) = 1;
    input(0, 1) = 6;
    input(0, 2) = 2;
    input(1, 0) = 5;
    input(1, 1) = 3;
    input(1, 2) = 1;
    input(2, 0) = 7;
    input(2, 1) = 0;
    input(2, 2) = 4;

    // Matrix<int> kernel = Matrix<int>(3, 2);
    // kernel(0, 0) = 1;
    // kernel(0, 1) = 2;
    // kernel(1, 0) = 3;
    // kernel(1, 1) = -1;
    // kernel(2, 0) = 0;
    // kernel(2, 1) = 2;
    // cout << kernel.ToString() << endl;

    Matrix<Matrix<int>> oui(1, 2);
    oui(0, 0) = input;
    oui(0, 1) = input.Flip();

    Matrix<Matrix<int>> non(2, 1);
    non(0, 0) = input.Flip();
    non(1, 0) = input;

    Matrix<Matrix<int>> tmp = oui * non;

    cout << oui.ToString() << endl;
    cout << non.ToString() << endl;
    cout << tmp.ToString() << endl;

    Matrix<int> res = input * input.Flip() + input.Flip() * input;
    cout << res.ToString() << endl;


    return 0;
}
