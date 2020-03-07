/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-06 21:52:43
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-07 11:54:32
 */
#include <iostream>
#include <ctime>
#include <Windows.h>
using namespace std;
#ifndef STRASSEN_HH
#define STRASSEN_HH
template <typename T>
class Strassen_class
{
public:
    void ADD(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize);
    void SUB(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize);
    void MUL(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize); //朴素算法实现
    void FillMatrix(T **MatrixA, T **MatrixB, int length);                //A,B矩阵赋值
    void PrintMatrix(T **MatrixA, int MatrixSize);                        //打印矩阵
    void Strassen(int N, T **MatrixA, T **MatrixB, T **MatrixC);          //Strassen算法实现
};
template <typename T>
void Strassen_class<T>::ADD(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize)
{
    for (int i = 0; i < MatrixSize; i++)
    {
        for (int j = 0; j < MatrixSize; j++)
        {
            MatrixResult[i][j] = MatrixA[i][j] + MatrixB[i][j];
        }
    }
}
template <typename T>
void Strassen_class<T>::SUB(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize)
{
    for (int i = 0; i < MatrixSize; i++)
    {
        for (int j = 0; j < MatrixSize; j++)
        {
            MatrixResult[i][j] = MatrixA[i][j] - MatrixB[i][j];
        }
    }
}
template <typename T>
void Strassen_class<T>::MUL(T **MatrixA, T **MatrixB, T **MatrixResult, int MatrixSize)
{
    for (int i = 0; i < MatrixSize; i++)
    {
        for (int j = 0; j < MatrixSize; j++)
        {
            MatrixResult[i][j] = 0;
            for (int k = 0; k < MatrixSize; k++)
            {
                MatrixResult[i][j] = MatrixResult[i][j] + MatrixA[i][k] * MatrixB[k][j];
            }
        }
    }
}

/*
c++使用二维数组，申请动态内存方法
申请
int **A;
A = new int *[desired_array_row];
for ( int i = 0; i < desired_array_row; i++)
     A[i] = new int [desired_column_size];

释放
for ( int i = 0; i < your_array_row; i++)
    delete [] A[i];
delete[] A;

*/
template <typename T>
void Strassen_class<T>::Strassen(int N, T **MatrixA, T **MatrixB, T **MatrixC)
{
    int n = 1;
    while (n < N)
        n *= 2;
    if (n <= 2) //分治门槛，小于这个值时不再进行递归计算，而是采用常规矩阵计算方法
    {
        MUL(MatrixA, MatrixB, MatrixC, N);
    }
    else
    {
        T **a, **b, **c;
        a = new T *[n];
        b = new T *[n];
        c = new T *[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = new T[n];
            b[i] = new T[n];
            c[i] = new T[n];
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                c[i][j] = 0;
                if (i < N && j < N)
                {
                    a[i][j] = MatrixA[i][j];
                    b[i][j] = MatrixB[i][j];
                }
                else
                {
                    a[i][j] = 0;
                    b[i][j] = 0;
                }
            }
        }
        T **A11, **A12, **A21, **A22;
        T **B11, **B12, **B21, **B22;
        T **C11, **C12, **C21, **C22;
        T **M1;
        T **M2;
        T **M3;
        T **M4;
        T **M5;
        T **M6;
        T **M7;
        T **AResult;
        T **BResult;
        int HalfSize = n / 2;
        int newSize = n / 2;
        //making a 1 diminsional pointer based array.
        A11 = new T *[newSize];
        A12 = new T *[newSize];
        A21 = new T *[newSize];
        A22 = new T *[newSize];

        B11 = new T *[newSize];
        B12 = new T *[newSize];
        B21 = new T *[newSize];
        B22 = new T *[newSize];

        C11 = new T *[newSize];
        C12 = new T *[newSize];
        C21 = new T *[newSize];
        C22 = new T *[newSize];

        M1 = new T *[newSize];
        M2 = new T *[newSize];
        M3 = new T *[newSize];
        M4 = new T *[newSize];
        M5 = new T *[newSize];
        M6 = new T *[newSize];
        M7 = new T *[newSize];

        AResult = new T *[newSize];
        BResult = new T *[newSize];

        //making that 1 diminsional pointer based array , a 2D pointer based array
        for (int i = 0; i < newSize; i++)
        {
            A11[i] = new T[newSize];
            A12[i] = new T[newSize];
            A21[i] = new T[newSize];
            A22[i] = new T[newSize];

            B11[i] = new T[newSize];
            B12[i] = new T[newSize];
            B21[i] = new T[newSize];
            B22[i] = new T[newSize];

            C11[i] = new T[newSize];
            C12[i] = new T[newSize];
            C21[i] = new T[newSize];
            C22[i] = new T[newSize];

            M1[i] = new T[newSize];
            M2[i] = new T[newSize];
            M3[i] = new T[newSize];
            M4[i] = new T[newSize];
            M5[i] = new T[newSize];
            M6[i] = new T[newSize];
            M7[i] = new T[newSize];

            AResult[i] = new T[newSize];
            BResult[i] = new T[newSize];
        }
        //splitting input Matrixes, into 4 submatrices each.
        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < n / 2; j++)
            {
                A11[i][j] = a[i][j];
                A12[i][j] = a[i][j + n / 2];
                A21[i][j] = a[i + n / 2][j];
                A22[i][j] = a[i + n / 2][j + n / 2];

                B11[i][j] = b[i][j];
                B12[i][j] = b[i][j + n / 2];
                B21[i][j] = b[i + n / 2][j];
                B22[i][j] = b[i + n / 2][j + n / 2];
            }
        }

        //here we calculate M1..M7 matrices .
        //M1[][]
        ADD(A11, A22, AResult, HalfSize);
        ADD(B11, B22, BResult, HalfSize);         //p5=(a+d)*(e+h)
        Strassen(HalfSize, AResult, BResult, M1); //now that we need to multiply this , we use the strassen itself .

        //M2[][]
        ADD(A21, A22, AResult, HalfSize);     //M2=(A21+A22)B11   p3=(c+d)*e
        Strassen(HalfSize, AResult, B11, M2); //Mul(AResult,B11,M2);

        //M3[][]
        SUB(B12, B22, BResult, HalfSize);     //M3=A11(B12-B22)   p1=a*(f-h)
        Strassen(HalfSize, A11, BResult, M3); //Mul(A11,BResult,M3);

        //M4[][]
        SUB(B21, B11, BResult, HalfSize);     //M4=A22(B21-B11)    p4=d*(g-e)
        Strassen(HalfSize, A22, BResult, M4); //Mul(A22,BResult,M4);

        //M5[][]
        ADD(A11, A12, AResult, HalfSize);     //M5=(A11+A12)B22   p2=(a+b)*h
        Strassen(HalfSize, AResult, B22, M5); //Mul(AResult,B22,M5);

        //M6[][]
        SUB(A21, A11, AResult, HalfSize);
        ADD(B11, B12, BResult, HalfSize);         //M6=(A21-A11)(B11+B12)   p7=(c-a)(e+f)
        Strassen(HalfSize, AResult, BResult, M6); //Mul(AResult,BResult,M6);

        //M7[][]
        SUB(A12, A22, AResult, HalfSize);
        ADD(B21, B22, BResult, HalfSize);         //M7=(A12-A22)(B21+B22)    p6=(b-d)*(g+h)
        Strassen(HalfSize, AResult, BResult, M7); //Mul(AResult,BResult,M7);

        //C11 = M1 + M4 - M5 + M7;
        ADD(M1, M4, AResult, HalfSize);
        SUB(M7, M5, BResult, HalfSize);
        ADD(AResult, BResult, C11, HalfSize);

        //C12 = M3 + M5;
        ADD(M3, M5, C12, HalfSize);

        //C21 = M2 + M4;
        ADD(M2, M4, C21, HalfSize);

        //C22 = M1 + M3 - M2 + M6;
        ADD(M1, M3, AResult, HalfSize);
        SUB(M6, M2, BResult, HalfSize);
        ADD(AResult, BResult, C22, HalfSize);

        //at this point , we have calculated the c11..c22 matrices, and now we are going to
        //put them together and make a unit matrix which would describe our resulting Matrix.
        //组合小矩阵到一个大矩阵
        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < n / 2; j++)
            {
                c[i][j] = C11[i][j];
                c[i][j + n / 2] = C12[i][j];
                c[i + n / 2][j] = C21[i][j];
                c[i + n / 2][j + n / 2] = C22[i][j];
            }
        }
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                MatrixC[i][j] = c[i][j];
            }
        }
        // 释放矩阵内存空间
        for (int i = 0; i < n; i++)
            delete[] a[i], b[i], c[i];
        delete[] a, b, c;
        for (int i = 0; i < newSize; i++)
        {
            delete[] A11[i], A12[i], A21[i], A22[i];

            delete[] B11[i], B12[i], B21[i], B22[i];
            delete[] C11[i], C12[i], C21[i], C22[i];
            delete[] M1[i];
            delete[] M2[i];
            delete[] M3[i];
            delete[] M4[i];
            delete[] M5[i];
            delete[] M6[i];
            delete[] M7[i];
            delete[] AResult[i], BResult[i];
        }
        delete[] A11, A12, A21, A22;
        delete[] B11, B12, B21, B22;
        delete[] C11, C12, C21, C22;
        delete[] M1, M2, M3, M4, M5, M6, M7;
        delete[] AResult, BResult;

    } //end of else
}

template <typename T>
void Strassen_class<T>::FillMatrix(T **MatrixA, T **MatrixB, int length)
{
    for (int row = 0; row < length; row++)
    {
        for (int column = 0; column < length; column++)
        {

            MatrixB[row][column] = (MatrixA[row][column] = rand() % 5);
            //matrix2[row][column] = rand() % 2;//ba hazfe in khat 50% afzayeshe soorat khahim dasht
        }
    }
}
template <typename T>
void Strassen_class<T>::PrintMatrix(T **MatrixA, int MatrixSize)
{
    cout << endl;
    for (int row = 0; row < MatrixSize; row++)
    {
        for (int column = 0; column < MatrixSize; column++)
        {

            cout << MatrixA[row][column] << "\t";
            if ((column + 1) % ((MatrixSize)) == 0)
                cout << endl;
        }
    }
    cout << endl;
}
#endif

int main()
{
    Strassen_class<int> stra; //定义Strassen_class类对象
    int MatrixSize = 0;

    int **MatrixA; //存放矩阵A
    int **MatrixB; //存放矩阵B
    int **MatrixC; //存放结果矩阵

    clock_t startTime_For_Strassen;
    clock_t endTime_For_Strassen;
    srand(time(0));

    cout << "Please enter matrix dimensions N , we Will randomly generate an N x N matrix" << endl;
    cin >> MatrixSize;

    int N = MatrixSize; //for readiblity.

    //申请内存
    MatrixA = new int *[MatrixSize];
    MatrixB = new int *[MatrixSize];
    MatrixC = new int *[MatrixSize];

    for (int i = 0; i < MatrixSize; i++)
    {
        MatrixA[i] = new int[MatrixSize];
        MatrixB[i] = new int[MatrixSize];
        MatrixC[i] = new int[MatrixSize];
    }

    stra.FillMatrix(MatrixA, MatrixB, MatrixSize); //矩阵赋值

    //*******************Strassen multiplication test
    startTime_For_Strassen = clock();
    stra.Strassen(N, MatrixA, MatrixB, MatrixC); //strassen矩阵相乘算法
    endTime_For_Strassen = clock();
    stra.PrintMatrix(MatrixA, MatrixSize);
    stra.PrintMatrix(MatrixB, MatrixSize);
    stra.PrintMatrix(MatrixC, MatrixSize);

    cout << (endTime_For_Strassen - startTime_For_Strassen) << " Clocks.." << (endTime_For_Strassen - startTime_For_Strassen) / CLOCKS_PER_SEC << " Sec\n";
    system("Pause");
    return 0;
}