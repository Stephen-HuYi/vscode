/*
 * @Description: insertionsort
 * @Author: HuYi
 * @Date: 2020-03-22 16:41:56
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-23 21:15:52
 */

#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;

void insertionsort(int a[], int length)
{
    int temp;
    for (int i = 1; i < length; i++)
    {
        for (int j = i - 1; j >= 0 && a[j + 1] < a[j]; j--)
        {
            temp = a[j];
            a[j] = a[j + 1];
            a[j + 1] = temp;
        }
    }
}

int main()
{
    int *a;
    int N;
    cout << "Please enter the number of numbers to be sorted" << endl;
    cin >> N;
    a = new int[N];
    for (int i = 0; i < N; i++)
        a[i] = (rand() << 16) | (rand());
    clock_t startTime, endTime;
    startTime = clock(); //计时开始
    insertionsort(a, N);
    endTime = clock(); //计时结束
    cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "print result or not?(1 = yes / 0 = no)" << endl;
    int print;
    cin >> print;
    if (print)
    {
        cout << "result:" << endl;
        for (int i = 0; i < N; i++)
        {
            cout << a[i] << endl;
        }
    }
    delete[] a;
    cout << endl;
    system("pause");
    return 0;
}
