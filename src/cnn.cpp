#include <cstdlib>
#include <iostream>

#include "matrix/matrix.hpp"

using namespace std;

int main(void)
{
    Matrix *oui = new Matrix(3, 3);
    for (size_t y = 0; y < oui->height; y++)
    {
        for (size_t x = 0; x < oui->width; x++)
        {
            oui->data[y][x] = x + y;
        }
    }
    cout << oui->toString() << endl;
    delete oui;
}
