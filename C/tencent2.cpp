/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-04-26 19:59:33
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-26 21:11:16
 */
#include <iostream>
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
    int N, K;
    cin >> N >> K;
    int *len = new int[N];
    int **A;
    A = new int *[N];
    int cnt = 0;
    for (int i = 0; i < N; i++)
    {
        cin >> len[i];
        cnt += len[i];
        A[i] = new int[len[i]];
        for (int j = 0; j < len[i]; j++)
        {
            cin >> A[i][j];
        }
    }

    int *array = new int[cnt];
    int cnt2 = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < len[i]; j++)
        {
            array[cnt2 + j] = A[i][j];
        }
        cnt2 += len[i];
    }
    quicksort(array, 0, cnt - 1);
    for (int i = 0; i < K - 1; i++)
    {
        cout << array[cnt - 1 - i] << " ";
    }
    cout << array[cnt - K];

    system("pause");
    return 0;
}