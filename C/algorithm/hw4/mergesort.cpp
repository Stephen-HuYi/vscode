/*
 * @Description:mergesort
 * @Author: HuYi
 * @Date: 2020-03-22 16:42:04
 * @LastEditors: HuYi
 * @LastEditTime: 2020-03-23 21:29:49
 */

#include <iostream>
#include <windows.h>
#include <ctime>
using namespace std;

void mergeArray(int arr[], int left, int mid, int right, int temp[])
{
    int i = left;
    int j = mid + 1;
    int m = mid;
    int r = right;
    int t = 0;
    while (i <= m && j <= r)
    {
        if (arr[i] <= arr[j])
            temp[t++] = arr[i++];
        else
            temp[t++] = arr[j++];
    }
    while (i <= m)
        temp[t++] = arr[i++];
    while (j <= r)
        temp[t++] = arr[j++];
    for (i = 0; i < t; i++)
        arr[left + i] = temp[i];
}
void mergesort(int arr[], int left, int right, int temp[])
{
    if (left < right)
    {
        int mid = (left + right) / 2;
        mergesort(arr, left, mid, temp);
        mergesort(arr, mid + 1, right, temp);
        mergeArray(arr, left, mid, right, temp);
    }
}

int main()
{
    int *a;
    int N;
    cout << "Please enter the number of numbers to be sorted" << endl;
    cin >> N;
    a = new int[N];
    int *p = new int[N];
    for (int i = 0; i < N; i++)
        a[i] = (rand() << 16) | (rand());
    clock_t startTime, endTime;
    startTime = clock(); //计时开始
    mergesort(a, 0, N - 1, p);
    endTime = clock(); //计时结束
    delete[] p;
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
