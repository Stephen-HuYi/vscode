/*
 * @Description:quicksort 
 * @Author: HuYi
 * @Date: 2020-03-22 16:41:48
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-01 15:32:41
 */

#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;

void quicksort(int a[], int left, int right)
{
    int i = left;
    int j = right;
    int temp = a[i];
    if (i < j)
    {
        while (i < j)
        {
            while (i < j && a[j] >= temp)
                j--;
            if (i < j)
            {
                a[i++] = a[j];
            }
            while (i < j && a[i] < temp)
                i++;
            if (i < j)
            {
                a[j--] = a[i];
            }
        }
        a[i] = temp;
        quicksort(a, left, i - 1);
        quicksort(a, i + 1, right);
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
    quicksort(a, 0, N - 1);
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
