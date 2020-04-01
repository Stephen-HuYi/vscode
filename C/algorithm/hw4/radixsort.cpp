/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-03-22 16:42:13
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-01 15:34:09
 */

#include <iostream>
#include <windows.h>
#include <ctime>

using namespace std;

int digit(int num) //获取num的位数
{
    return num < 10 ? 1 : 1 + digit(num / 10);
}

int getnum(int num, int pos)
{
    int temp = 1;
    for (int i = 0; i < pos; i++)
    {
        temp *= 10;
    }
    return (num / temp) % 10;
}

void radixsort(int a[], int N, int max)
{
    int *b[10];  //用来存放桶
    int cnt[10]; //用来记录每个桶中数字个数
    for (int i = 0; i < 10; i++)
    {
        b[i] = new int[N];
        cnt[i] = 0;
    }
    int d = digit(max);
    for (int dd = 0; dd < d; dd++) //调用稳定的排序算法：桶排序，从低位向高位对每一位进行排序
    {
        for (int i = 0; i < N; i++)
        {
            int num = getnum(a[i], dd); //获取第dd位的数字
            b[num][cnt[num]++] = a[i];
        }
        int n = 0;
        for (int i = 0; i < 10; i++) //只需将桶合并：因为十个桶，每个桶中的那位数字都一样，所以按照桶分好后不需要再排序
        {
            for (int j = 0; j < cnt[i]; j++)
            {
                a[n++] = b[i][j];
            }
            cnt[i] = 0;
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
    int max = 0;
    for (int i = 0; i < N; i++) //随机生成N个数并求出最大值
    {
        a[i] = (rand() << 16) | (rand());
        if (max < a[i])
            max = a[i];
    }
    clock_t startTime, endTime;
    startTime = clock(); //计时开始
    radixsort(a, N, max);
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