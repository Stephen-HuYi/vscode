/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-06 08:32:14
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-12 10:48:39
 */
#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;

void Print(int **A, int N)
{
    cout << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main()
{
    cout << "Please enter matrix dimensions N , we Will randomly generate an N x N matrix" << endl;
    int N;
    cin >> N;
    //动态分配内存空间
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
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    }
    clock_t startTime, endTime;
    startTime = clock(); //计时开始
    //直接暴力算法O(n^3)计算矩阵乘法
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    endTime = clock(); //计时结束
    cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "print matrixs?(1 = yes / 0 = no)" << endl;
    int print;
    cin >> print;
    if (print)
    {
        Print(A, N);
        Print(B, N);
        Print(C, N);
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