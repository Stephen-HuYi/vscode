/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-04-26 19:59:49
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-26 21:01:56
 */
#include <iostream>
using namespace std;
int Fun(int n)
{
    int count = 0;
    while (n > 4)
    {
        count = count + n / 5;
        n /= 5;
    }
    return count;
}
int main()
{
    int N;
    cin >> N;
    int *a = new int[N];
    for (int i = 0; i < N; i++)
    {
        cin >> a[i];
    }
    for (int i = 0; i < N; i++)
    {
        cout << a[i];
    }
    system("pause");
    return 0;
}