/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-06-02 11:31:45
 * @LastEditors: HuYi
 * @LastEditTime: 2020-06-02 18:38:10
 */
#include <iostream>
#include <windows.h>
using namespace std;

int m, n;		   //定义参数m、n为全局变量
DWORD Write, Read; //定义管道传递用到的参数
HANDLE fWrite, fRead, gWrite, gRead, FWrite, FRead;

DWORD WINAPI f(LPVOID lpParam) //f函数
{
	int *s = new int;
	*s = 1;
	for (int i = 1; i <= m; i++) //计算f函数，即阶乘
		*s *= i;
	WriteFile(fWrite, s, sizeof(int), &Write, NULL);
	CloseHandle(fWrite);
	delete s;
	return 0;
}
DWORD WINAPI g(LPVOID lpParam) //g函数
{
	int *b = new int;
	*b = 1;
	int a = 0, c;
	for (int i = 1; i < n; i++) //计算g函数，即斐波那契数列
	{
		c = a + *b;
		a = *b;
		*b = c;
	}
	WriteFile(gWrite, b, sizeof(int), &Write, NULL);
	CloseHandle(gWrite);
	delete b;
	return 0;
}
DWORD WINAPI F(LPVOID lpParam) //F函数
{
	int *f = new int;
	int *g = new int;
	int *F = new int;
	//从fRead和gRead中读取f函数和g函数的结果
	ReadFile(fRead, f, sizeof(int), &Read, NULL);
	ReadFile(gRead, g, sizeof(int), &Read, NULL);
	*F = *f + *g;
	WriteFile(FWrite, F, sizeof(int), &Write, NULL);
	return 0;
}

int main()
{
	cout << "Input m and n:";
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
	cout << "F(" << m << "," << n << ")"
		 << "=" << *F << endl;
	Sleep(2000);
	CloseHandle(fThread); //关闭线程
	CloseHandle(gThread);
	CloseHandle(FThread);
	return 0;
}