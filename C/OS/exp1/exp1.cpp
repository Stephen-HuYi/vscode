#include <iostream>
#include <fstream>
#include <windows.h>
#include <queue>
using namespace std;

#define N 3                                           //设置柜台数
#define M 100                                         //设顾客数最大为100
int time = 0;                                         //记录时间
int customer_amount = 0;                              //记录顾客数
int num_out = 0;                                      //记录离开银行的顾客数
queue<int> counter;                                   //空闲柜台队列
HANDLE Thread[M], ThreadP, ThreadV;                   //柜台线程与P、V线程
HANDLE Mutex = CreateMutex(NULL, FALSE, NULL);        //顾客拿号、柜台叫号的互斥量
HANDLE Semaphore = CreateSemaphore(NULL, N, N, NULL); //用于实现银行职员进程同步的信号量

//用一个结构体来表示顾客
struct customer
{
    int customer_num, counter_num; //顾客序号和服务员柜号
    int time_in, time_out;         //顾客进入和离开银行的时间
    int time_start, time_serve;    //柜台开始服务的时间和顾客需要服务的时间
    customer *next;                //下一客户
};
customer *head = new customer;   //顾客链表的头指针
customer *current = head;        //顾客链表的当前指针
queue<customer *> customer_wait; //顾客等待队列

//顾客接受柜台服务过程
void Serve(customer *c)
{
    Sleep(1000 * c->time_serve);
    if (WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
    {
        counter.push(c->counter_num);
        ReleaseSemaphore(Semaphore, 1, NULL);
        cout << c->time_in << " " << c->time_start << " " << c->time_out << " " << c->counter_num << endl;
        ReleaseMutex(Mutex);
        num_out++;
    }
}

//P函数，柜台进行叫号
void P()
{
    while (1)
    {
        if (customer_wait.size() == 0)
            continue;
        if (WaitForSingleObject(Semaphore, INFINITE) == WAIT_OBJECT_0 && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            customer *c = customer_wait.front();
            customer_wait.pop();
            c->counter_num = counter.front();
            counter.pop();
            c->time_start = time;
            c->time_out = time + c->time_serve;
            LPVOID lpParamter = c;
            Thread[c->counter_num - 1] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(Serve), lpParamter, 0, NULL);
            ReleaseMutex(Mutex);
        }
    }
};

//V函数，顾客进入银行
void V()
{
    while (1)
    {
        if (current->next == NULL)
            continue;
        if (current->next->time_in == time && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            customer_wait.push(current->next);
            current = current->next;
            ReleaseMutex(Mutex);
        }
    }
}

//主函数
int main()
{
    //柜台号入队列
    for (int i = 1; i <= N; i++)
    {
        counter.push(i);
    }
    //读文件得到顾客数据
    ifstream f;
    f.open("input.txt");
    if (!f)
    {
        cout << "Open Error!" << endl;
        exit(1);
    }
    head->next = NULL;
    do
    {
        customer *p = head;
        customer *q = new customer;
        f >> q->customer_num;
        f >> q->time_in;
        f >> q->time_serve;
        while (p->next != NULL && p->next->time_in <= q->time_in)
        {
            p = p->next;
        }
        q->next = p->next;
        p->next = q;
        customer_amount++;
    } while (!f.eof());
    f.close();
    //开启P、V线程
    ThreadP = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(P), NULL, 0, NULL);
    ThreadV = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(V), NULL, 0, NULL);
    //时间依次+1，直到顾客全部离开
    while (1)
    {
        Sleep(1000);
        time++;
        if (num_out == customer_amount)
        {
            break;
        }
    }
    //关闭线程和信号量
    for (int i = 0; i < N; i++)
    {
        CloseHandle(Thread[i]);
    }
    CloseHandle(ThreadP);
    CloseHandle(ThreadV);
    CloseHandle(Mutex);
    CloseHandle(Semaphore);
    system("pause");
    return 0;
}