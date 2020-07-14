/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-06-02 11:31:45
 * @LastEditors: HuYi
 * @LastEditTime: 2020-06-05 21:23:53
 */
#include <iostream>
#include <windows.h>
using namespace std;

//定义全局变量，包含参数m、n和管道传递的相关参数
int m, n;
DWORD Write, Read;
HANDLE fWrite, fRead, gWrite, gRead, FWrite, FRead;

DWORD WINAPI f(LPVOID lpParam) //f函数
{
	int *f = new int;
	*f = 1;
	for (int i = 1; i <= m; i++) //计算f函数，即阶乘
		*f *= i;
	WriteFile(fWrite, f, sizeof(int), &Write, NULL);
	CloseHandle(fWrite);
	delete f;
	return 0;
}
DWORD WINAPI g(LPVOID lpParam) //g函数
{
	int *g = new int;
	*g = 1;
	int a = 0, b;
	for (int i = 1; i < n; i++) //计算g函数，即斐波那契数列
	{
		b = a + *g;
		a = *g;
		*g = b;
	}
	WriteFile(gWrite, g, sizeof(int), &Write, NULL);
	CloseHandle(gWrite);
	delete g;
	return 0;
}
DWORD WINAPI F(LPVOID lpParam) //F函数
{
	int *f = new int;
	int *g = new int;
	int *F = new int;
	//读取f和g的值并求和得到F
	ReadFile(fRead, f, sizeof(int), &Read, NULL);
	ReadFile(gRead, g, sizeof(int), &Read, NULL);
	*F = *f + *g;
	WriteFile(FWrite, F, sizeof(int), &Write, NULL);
	delete f, g, F;
	return 0;
}

int main()
{
	cout << "Please input m and n:";
	cin >> m >> n;			//输入参数m和n
	SECURITY_ATTRIBUTES sa; //建立f、g、F函数的管道及相关参数
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = NULL;
	sa.bInheritHandle = TRUE;
	CreatePipe(&fRead, &fWrite, &sa, 0);
	CreatePipe(&gRead, &gWrite, &sa, 0);
	CreatePipe(&FRead, &FWrite, &sa, 0);
	//建立f、g、F函数线程
	HANDLE fThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)f, NULL, 0, NULL);
	HANDLE gThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)g, NULL, 0, NULL);
	HANDLE FThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)F, NULL, 0, NULL);
	//读取F函数结果并输出
	int *F = new int;
	ReadFile(FRead, F, sizeof(int), &Read, NULL);
	cout << "Result:F(" << m << "," << n << ")"
		 << "=" << *F << endl;
	delete F;
	Sleep(5000);
	CloseHandle(fThread); //结束线程
	CloseHandle(gThread);
	CloseHandle(FThread);
	return 0;
}