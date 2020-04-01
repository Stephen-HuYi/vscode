/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-22 16:41:40
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-01 09:57:10
 */

#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;
void shellsort(int a[], int N)
{
    int temp;
    for (int inc = N / 2; inc > 0; inc /= 2)
    {
        for (int i = inc; i < N; i++)
        {
            int j = i - inc;
            while (j >= 0 && a[j] > a[j + inc])
            {
                temp = a[j + inc];
                a[j + inc] = a[j];
                a[j] = temp;
                j -= inc;
            }
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
    shellsort(a, N);
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
