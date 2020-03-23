/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-22 16:41:40
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-23 21:51:30
 */

#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;
void shellsort(int a[], int N)
{
    int s;
    s = N / 2;
    if (s % 2 == 0)
    {
        s++;
    }
    while (s > 0)
    {
        for (int j = s; j < N; j++)
        {
            int i = j - s;
            int temp = a[j];
            while (i >= 0 && a[i] > temp)
            {
                a[i + s] = a[i];
                i = i - s;
            }
            if (i != j - s)
                a[i + s] = temp;
        }
        if (s == 1)
            break;
        s /= 2;
        if (s % 2 == 0)
            s++;
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
