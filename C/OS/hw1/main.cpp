#include <iostream>
#include <fstream>
#include <windows.h>
#include <queue>
using namespace std;

#define N 3                                           //柜台数设为3
#define M 100                                         //设顾客数最大为100
#define period 1000                                   //定义1s为1000ms
int time = 0;                                         //时间
int number = 0;                                       //顾客数
int out = 0;                                          //离开银行的顾客数
queue<int> Staff;                                     //柜台队列
HANDLE Mutex = CreateSemaphore(NULL, 1, 1, NULL);     //顾客拿号、柜台叫号的互斥量
HANDLE Semaphore = CreateSemaphore(NULL, N, N, NULL); //用于实现银行职员进程同步的信号量
//柜台线程与P、V线程
HANDLE Thread[M], ThreadP, ThreadV;
DWORD ThreadAdd[M], ThreadPAdd, ThreadVAdd;
//顾客
struct Custom
{
    int CustomNumber; //顾客序号
    int EnterTime;    //顾客进入银行的时间
    int ServeTime;    //顾客需要服务的时间
    int StartTime;    //柜台开始服务的时间
    int LeaveTime;    //顾客离开银行的时间
    int StaffNumber;  //服务柜台号
    Custom *Next;     //下一客户
};
Custom *head = new Custom;  //客户链表头指针
Custom *now = head;         //客户链表当前指针
queue<Custom *> CustomWait; //等待中的顾客队列
//顾客接受柜台服务过程
void Service(Custom *c)
{
    Sleep(period * c->ServeTime);
    if (WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
    {
        Staff.push(c->StaffNumber);
        ReleaseSemaphore(Semaphore, 1, NULL);
        cout << c->EnterTime << ends << c->StartTime << ends << c->LeaveTime << ends << c->StaffNumber << endl;
        ReleaseSemaphore(Mutex, 1, NULL);
        out++;
    }
}
//P函数，柜台进行叫号
void P()
{
    while (1)
    {
        if (CustomWait.size() == 0)
            continue;
        if (WaitForSingleObject(Semaphore, INFINITE) == WAIT_OBJECT_0 && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            Custom *c = CustomWait.front();
            CustomWait.pop();
            c->StaffNumber = Staff.front();
            Staff.pop();
            c->StartTime = time;
            c->LeaveTime = c->StartTime + c->ServeTime;
            LPVOID lparam = c;
            Thread[c->StaffNumber - 1] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(Service), lparam, 0, &ThreadAdd[c->StaffNumber - 1]);
            ReleaseSemaphore(Mutex, 1, NULL);
        }
    }
};
//V函数，顾客进入银行
void V()
{
    while (1)
    {
        if (now->Next == NULL)
            continue;
        if (now->Next->EnterTime == time && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            CustomWait.push(now->Next);
            now = now->Next;
            ReleaseSemaphore(Mutex, 1, NULL);
        }
    }
}
//主程序
int main()
{
    //输入顾客数据
    ifstream InFile;
    InFile.open("test.txt");
    head->Next = NULL;
    do
    {
        Custom *p = head;
        Custom *q = new Custom;
        InFile >> q->CustomNumber;
        InFile >> q->EnterTime;
        InFile >> q->ServeTime;
        while (p->Next != NULL && p->Next->EnterTime <= q->EnterTime)
            p = p->Next;
        q->Next = p->Next;
        p->Next = q;
        number++;
    } while (!InFile.eof());
    //将柜台号入队列
    for (int i = 1; i <= N; i++)
        Staff.push(i);
    //开启P、V两个线程
    ThreadP = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(P), NULL, 0, &ThreadPAdd);
    ThreadV = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(V), NULL, 0, &ThreadVAdd);
    //时间依次+1，直到顾客全部离开
    while (1)
    {
        Sleep(period);
        time++;
        if (out == number)
            break;
    }
    //关闭线程
    for (int i = 0; i < N; i++)
        CloseHandle(Thread[i]);
    CloseHandle(ThreadP);
    CloseHandle(ThreadV);
    system("pause");
    return 0;
}