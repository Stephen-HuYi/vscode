/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-06 08:32:14
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-07 12:09:06
 */
#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;
template <typename T>
class Mat
{
public:
    void ADD(T **A, T **B, T **Res, int Size);
    void SUB(T **A, T **B, T **Res, int Size);
    void MUL(T **A, T **B, T **Res, int Size); //朴素算法实现
    void Fill(T **A, T **B, int Len);          //A,B矩阵赋值
    void Print(T **A, int Size);               //打印矩阵
    void Strassen(int N, T **A, T **B, T **C); //strassen算法实现
};
template <typename T>
void Mat<T>::ADD(T **A, T **B, T **Res, int Size)
{
    for (int i = 0; i < Size; i++)
    {
        for (int j = 0; j < Size; j++)
        {
            Res[i][j] = A[i][j] + B[i][j];
        }
    }
}
template <typename T>
void Mat<T>::SUB(T **A, T **B, T **Res, int Size)
{
    for (int i = 0; i < Size; i++)
    {
        for (int j = 0; j < Size; j++)
        {
            Res[i][j] = A[i][j] - B[i][j];
        }
    }
}
template <typename T>
void Mat<T>::MUL(T **A, T **B, T **Res, int Size)
{
    for (int i = 0; i < Size; i++)
    {
        for (int j = 0; j < Size; j++)
        {
            Res[i][j] = 0;
            for (int k = 0; k < Size; k++)
            {
                Res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
template <typename T>
void Mat<T>::Strassen(int N, T **A, T **B, T **C)
{
    //将矩阵维度扩充为2的幂次
    int n = 1;
    while (n < N)
        n *= 2;
    if (n <= 128)
    {
        MUL(A, B, C, N);
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
                    a[i][j] = A[i][j];
                    b[i][j] = B[i][j];
                }
                else
                {
                    a[i][j] = 0;
                    b[i][j] = 0;
                }
            }
        }
        T **a11, **a12, **a21, **a22;
        T **b11, **b12, **b21, **b22;
        T **c11, **c12, **c21, **c22;
        T **m1, **m2, **m3, **m4, **m5, **m6, **m7;
        T **resa, **resb;
        int halfsize = n / 2;
        a11 = new T *[halfsize];
        a12 = new T *[halfsize];
        a21 = new T *[halfsize];
        a22 = new T *[halfsize];
        b11 = new T *[halfsize];
        b12 = new T *[halfsize];
        b21 = new T *[halfsize];
        b22 = new T *[halfsize];
        c11 = new T *[halfsize];
        c12 = new T *[halfsize];
        c21 = new T *[halfsize];
        c22 = new T *[halfsize];
        m1 = new T *[halfsize];
        m2 = new T *[halfsize];
        m3 = new T *[halfsize];
        m4 = new T *[halfsize];
        m5 = new T *[halfsize];
        m6 = new T *[halfsize];
        m7 = new T *[halfsize];
        resa = new T *[halfsize];
        resb = new T *[halfsize];
        for (int i = 0; i < halfsize; i++)
        {
            a11[i] = new T[halfsize];
            a12[i] = new T[halfsize];
            a21[i] = new T[halfsize];
            a22[i] = new T[halfsize];
            b11[i] = new T[halfsize];
            b12[i] = new T[halfsize];
            b21[i] = new T[halfsize];
            b22[i] = new T[halfsize];
            c11[i] = new T[halfsize];
            c12[i] = new T[halfsize];
            c21[i] = new T[halfsize];
            c22[i] = new T[halfsize];
            m1[i] = new T[halfsize];
            m2[i] = new T[halfsize];
            m3[i] = new T[halfsize];
            m4[i] = new T[halfsize];
            m5[i] = new T[halfsize];
            m6[i] = new T[halfsize];
            m7[i] = new T[halfsize];
            resa[i] = new T[halfsize];
            resb[i] = new T[halfsize];
        }
        //splitting input Matrixes, into 4 submatrices each.
        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < n / 2; j++)
            {
                a11[i][j] = a[i][j];
                a12[i][j] = a[i][j + n / 2];
                a21[i][j] = a[i + n / 2][j];
                a22[i][j] = a[i + n / 2][j + n / 2];
                b11[i][j] = b[i][j];
                b12[i][j] = b[i][j + n / 2];
                b21[i][j] = b[i + n / 2][j];
                b22[i][j] = b[i + n / 2][j + n / 2];
            }
        }
        //m1[][]
        ADD(a11, a22, resa, halfsize);
        ADD(b11, b22, resb, halfsize);
        Strassen(halfsize, resa, resb, m1);
        //m2[][]
        ADD(a21, a22, resa, halfsize);
        Strassen(halfsize, resa, b11, m2);
        //m3[][]
        SUB(b12, b22, resb, halfsize);
        Strassen(halfsize, a11, resb, m3);
        //m4[][]
        SUB(b21, b11, resb, halfsize);
        Strassen(halfsize, a22, resb, m4);
        //m5[][]
        ADD(a11, a12, resa, halfsize);
        Strassen(halfsize, resa, b22, m5);
        //m6[][]
        SUB(a21, a11, resa, halfsize);
        ADD(b11, b12, resb, halfsize);
        Strassen(halfsize, resa, resb, m6);
        //m7[][]
        SUB(a12, a22, resa, halfsize);
        ADD(b21, b22, resb, halfsize);
        Strassen(halfsize, resa, resb, m7);
        //C11 = M1 + M4 - M5 + M7;
        ADD(m1, m4, resa, halfsize);
        SUB(m7, m5, resb, halfsize);
        ADD(resa, resb, c11, halfsize);
        //C12 = M3 + M5;
        ADD(m3, m5, c12, halfsize);
        //C21 = M2 + M4;
        ADD(m2, m4, c21, halfsize);
        //C22 = M1 + M3 - M2 + M6;
        ADD(m1, m3, resa, halfsize);
        SUB(m6, m2, resb, halfsize);
        ADD(resa, resb, c22, halfsize);
        //at this point , we have calculated the c11..c22 matrices, and now we are going to
        //put them together and make a unit matrix which would describe our resulting Matrix.
        //组合小矩阵到一个大矩阵
        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < n / 2; j++)
            {
                c[i][j] = c11[i][j];
                c[i][j + n / 2] = c12[i][j];
                c[i + n / 2][j] = c21[i][j];
                c[i + n / 2][j + n / 2] = c22[i][j];
            }
        }
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i][j] = c[i][j];
            }
        }
        // 释放矩阵内存空间
        for (int i = 0; i < n; i++)
            delete[] a[i], b[i], c[i];
        for (int i = 0; i < halfsize; i++)
        {
            delete[] a11[i], a12[i], a21[i], a22[i];
            delete[] b11[i], b12[i], b21[i], b22[i];
            delete[] c11[i], c12[i], c21[i], c22[i];
            delete[] m1[i], m2[i], m3[i], m4[i], m5[i], m6[i], m7[i];
            delete[] resa[i], resb[i];
        }
        delete[] a, b, c;
        delete[] a11, a12, a21, a22;
        delete[] b11, b12, b21, b22;
        delete[] c11, c12, c21, c22;
        delete[] m1, m2, m3, m4, m5, m6, m7;
        delete[] resa, resb;
    }
}
template <typename T>
void Mat<T>::Fill(T **A, T **B, int len)
{
    for (int row = 0; row < len; row++)
    {
        for (int column = 0; column < len; column++)
        {
            A[row][column] = rand() % 5;
            B[row][column] = rand() % 5;
            //matrix2[row][column] = rand() % 2;//ba hazfe in khat 50% afzayeshe soorat khahim dasht
        }
    }
}
template <typename T>
void Mat<T>::Print(T **A, int Size)
{
    cout << endl;
    for (int row = 0; row < Size; row++)
    {
        for (int column = 0; column < Size; column++)
        {

            cout << A[row][column] << "\t";
            if ((column + 1) % ((Size)) == 0)
                cout << endl;
        }
    }
    cout << endl;
}

int main()
{
    cout << "Please enter matrix dimensions N , we Will randomly generate an N x N matrix" << endl;
    int N;
    cin >> N;
    Mat<int> mat; //定义Mat类对象
    int **A, **B, **C;
    A = new int *[N];
    B = new int *[N];
    C = new int *[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new int[N];
        B[i] = new int[N];
        C[i] = new int[N];
    }
    clock_t startTime, endTime;
    mat.Fill(A, B, N);
    startTime = clock();      //计时开始
    mat.Strassen(N, A, B, C); //strassen矩阵相乘算法
    endTime = clock();        //计时结束
    cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    //输出运算结果(矩阵维度较小时检验用)
    cout << "print matrixs?(1 = yes / 0 = no)" << endl;
    int print;
    cin >> print;
    if (print)
    {
        mat.Print(A, N);
        mat.Print(B, N);
        mat.Print(C, N);
    }
    //释放内存空间
    for (int i = 0; i < N; i++)
    {
        delete[] A[i], B[i], C[i];
    }
    delete[] A, B, C;
    system("pause");
    return 0;
}