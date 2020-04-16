/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-04-16 20:14:18
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-16 21:25:14
 */
#include <iostream>
#include <fstream>
#include <queue>
#include <windows.h>
using namespace std;
#define CLE 3                                             //柜员数
#define CUS 100                                           //最大顾客数
#define period 1000                                       //定义1s为1000ms
int time = 0;                                             //时间
int cun = 0;                                              //顾客数
int outn = 0;                                             //离开银行的顾客数
queue<int> Staff;                                         //柜台队列
HANDLE Mutex = CreateSemaphore(NULL, 1, 1, NULL);         //顾客取号、柜台叫号的互斥量
HANDLE Semaphore = CreateSemaphore(NULL, CLE, CLE, NULL); //用于实现银行职员进程同步的信号量

HANDLE Thread[CUS]; //柜台线程
DWORD ThreadAdd[CUS];

HANDLE ThreadP; //P、V线程
HANDLE ThreadV;
DWORD ThreadPAdd;
DWORD ThreadVAdd;

struct Custom //顾客
{
    int CustomNumber; //顾客序号
    int EnterTime;    //进入银行的时间
    int ServeTime;    //需要服务的时长
    int StartTime;    //柜台开始服务的时间
    int LeaveTime;    //离开银行的时间
    int StaffNumber;  //服务柜台号
    Custom *Next;     //下一客户
};
Custom *head = new Custom;  //客户链表头指针
Custom *now = head;         //客户链表当前指针
queue<Custom *> CustomWait; //等待中的顾客队列

void Service(Custom *c) //顾客接受柜台服务过程
{
    Sleep(period * c->ServeTime);
    if (WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
    {
        Staff.push(c->StaffNumber);
        ReleaseSemaphore(Semaphore, 1, NULL);
        cout << c->EnterTime << ' ' << c->StartTime << ' ' << c->LeaveTime << ' ' << c->StaffNumber << endl;
        ReleaseSemaphore(Mutex, 1, NULL);
        outn++;
    }
}

void P() //P函数，柜台进行叫号
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
}

void V() //V函数，顾客进入银行
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

int main()
{
    ifstream InFile; //输入顾客数据
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
        cun++;
    } while (!InFile.eof());

    for (int i = 1; i <= CLE; i++) //将柜台号入队列
        Staff.push(i);

    //开启P、V两个线程
    ThreadP = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(P), NULL, 0, &ThreadPAdd);
    ThreadV = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(V), NULL, 0, &ThreadVAdd);

    while (1) //时间依次+1，直到顾客全部离开
    {
        Sleep(period);
        time++;
        if (outn == cun)
            break;
    }

    for (int i = 0; i < CLE; i++) //关闭线程
        CloseHandle(Thread[i]);
    CloseHandle(ThreadP);
    CloseHandle(ThreadV);
    system("pause");
    return 0;
}