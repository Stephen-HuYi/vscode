/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-22 16:42:13
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-23 21:50:03
 */

#include <iostream>
#include <windows.h>
#include <ctime>
#include <queue>
using namespace std;
int digit_max(int num)
{
    int temp = num / 10;
    int cnt = 1;
    while (temp != 0)
    {
        temp = temp / 10;
        cnt++;
    }
    return cnt;
}

int get_max(int a[], int N)
{
    int max = 0;
    for (int i = 0; i < N; i++)
    {
        if (max < a[i])
        {
            max = a[i];
        }
    }
    return max;
}

int position(int num, int pos)
{
    int temp = 1;
    for (int i = 0; i < pos - 1; i++)
    {
        temp *= 10;
    }
    return (num / temp) % 10;
}

void radixsort(int a[], int N)
{
    int *radixarray[10];
    for (int i = 0; i < 10; i++)
    {
        radixarray[i] = (int *)malloc(sizeof(int) * (N + 1));
        radixarray[i][0] = 0;
    }
    int maxnum = get_max(a, N);
    int digit = digit_max(maxnum);
    for (int pos = 1; pos <= digit; pos++)
    {
        for (int i = 0; i < N; i++)
        {
            int num = position(a[i], pos);
            radixarray[num][0]++;
            int index = radixarray[num][0];
            radixarray[num][index] = a[i];
        }
        for (int i = 0, j = 0; i < 10; i++)
        {
            for (int k = 1; k <= radixarray[i][0]; k++)
            {
                a[j++] = radixarray[i][k];
            }
            radixarray[i][0] = 0;
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
    radixsort(a, N);
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