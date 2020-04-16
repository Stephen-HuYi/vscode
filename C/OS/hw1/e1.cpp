/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-04-16 20:16:22
 * @LastEditors: HuYi
 * @LastEditTime: 2020-04-16 22:43:31
 */
#include <iostream>
#include <fstream>
#include <windows.h>
using namespace std;
#define N 2 //银行职员的数量

int time = 0;
int num_out = 0;
int customernumber = 0;

HANDLE Mutex = CreateSemaphore(NULL, 1, 1, NULL);     //顾客取号、柜台叫号的互斥量
HANDLE Semaphore = CreateSemaphore(NULL, N, N, NULL); //用于实现银行职员进程同步的信号量
//柜台线程
HANDLE Thread[50];
DWORD ThreadAdd[50];
//P、V线程
HANDLE ThreadP;
HANDLE ThreadV;
DWORD ThreadPAdd;
DWORD ThreadVAdd;

class customer //顾客类
{
public:
    int num;       //顾客叫号
    int timein;    //来到银行的时间
    int timestart; //开始服务的时间
    int timeout;   //离开银行的时间
    int time;      //需要服务的时间
    int clerknum;  //服务的银行职员的编号
    customer *next, *following;
    customer()
    {
        next = NULL;
        following = NULL;
    }
};

class customerlist //顾客列表类
{
public:
    customer *first, *last;
    customerlist()
    {
        first = last = NULL;
    }
    void insert(customer *newman) //插入新结点
    {
        if (first == NULL)
        {
            first = last = newman;
        }
        else
        {
            last->following = newman;
            last = newman;
        }
    }
};

class queue //顾客队列
{
public:
    customer *early, *late;
    queue()
    {
        early = NULL;
        late = NULL;
    }
    void insert(customer *newman) //进入队列
    {
        if (early == NULL)
        {
            early = late = newman;
        }
        else
        {
            late->next = newman;
            late = newman;
        }
    }
    void remove() //出队
    {
        if (early == late)
        {
            early = late = NULL;
        }
        else
        {
            customer *p = early;
            early = early->next;
            p->next = NULL;
        }
    }
};

class clerk //银行职员类
{
public:
    bool busy;         //记录银行职员是否忙碌，忙碌为1，不忙则为0
    int worknum;       //银行职员编号
    int timeremaining; //银行职员剩余服务时间
    int customernum;   //当前服务的顾客编号
    clerk *next;
    clerk()
    {
        busy = 0;
        timeremaining = 0;
    }
    void clerkwork(queue *q, int time) //检查银行职员的工作情况
    {
        if (timeremaining != 0)
        {
            timeremaining = timeremaining - 1;
        }
        else
        {
            busy = 0;
            if (q->early != NULL) //叫一个新号
            {
                busy = 1;
                timeremaining = q->early->time - 1;
                customernum = q->early->num;
                q->early->timestart = time;
                q->early->timeout = q->early->time + time;
                q->early->clerknum = worknum;
                q->remove();
            }
        }
    }
};

class clerklist //银行职员列表类
{
public:
    clerk *first, *last;
    clerklist()
    {
        first = NULL;
        last = NULL;
    }
    void insert(clerk *newman) //进入队列
    {
        if (first == NULL)
        {
            first = last = newman;
        }
        else
        {
            last->next = newman;
            last = newman;
        }
    }
    void remove() //出队
    {
        if (first == last)
        {
            first->next = NULL;
            first = last = NULL;
        }
        else
        {
            clerk *p = first;
            first = first->next;
            p->next = NULL;
        }
    }
};

customerlist c; //顾客列表
clerklist staff;
clerk c1[N];
queue q; //等待队列
customer *current;

void work(customer *cc) //服务过程函数
{
    Sleep((cc->time) * 1000);
    if (WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
    {
        staff.insert(&c1[cc->clerknum]);
        ReleaseSemaphore(Semaphore, 1, NULL);
        cout << cc->timein << " " << cc->timestart << " " << cc->timeout << " " << cc->clerknum << endl;
        ReleaseSemaphore(Mutex, 1, NULL);
        num_out++;
    }
};

void P() //P函数
{
    while (1)
    {
        if (q.early == NULL)
            continue;
        if (WaitForSingleObject(Semaphore, INFINITE) == WAIT_OBJECT_0 && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            customer *cc = q.early;
            q.remove();
            cc->clerknum = staff.first->worknum;
            staff.remove();
            cc->timestart = time;
            cc->timeout = time + cc->time;
            LPVOID lparam = cc;
            Thread[cc->clerknum] = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(work), lparam, 0, &ThreadAdd[cc->clerknum]);
            ReleaseSemaphore(Mutex, 1, NULL);
        }
    }
};

void V() //V函数
{
    while (1)
    {
        if (current == NULL)
            continue;
        if (current->timein == time && WaitForSingleObject(Mutex, INFINITE) == WAIT_OBJECT_0)
        {
            q.insert(current);
            current = current->following;
            ReleaseSemaphore(Mutex, 1, NULL);
        }
    }
};

int main()
{
    int j;
    for (j = 0; j < N; j++)
    {
        c1[j].worknum = j; //录入银行职员的编号
        staff.insert(&c1[j]);
    }
    ifstream infile;
    infile.open("input.txt", ios::in); //打开测试文件
    if (!infile)
    {
        cerr << "Open Error!" << endl;
        exit(1);
    }
    while (infile.peek() != EOF) //读入每个顾客的数据
    {
        customer *newman = new customer();
        infile >> newman->num;
        infile.seekg(sizeof(char), ios::cur);
        infile >> newman->timein;
        infile.seekg(sizeof(char), ios::cur);
        infile >> newman->time;
        infile.seekg(sizeof(char), ios::cur);
        c.insert(newman);
        customernumber = customernumber + 1;
    }
    current = c.first;
    //开启P、V两个线程
    ThreadP = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(P), NULL, 0, &ThreadPAdd);
    ThreadV = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(V), NULL, 0, &ThreadVAdd);

    //时间依次+1，直到顾客全部离开
    while (1)
    {
        Sleep(1000);
        time++;
        if (num_out == customernumber)
            break;
    }

    //关闭线程
    for (int i = 0; i < N; i++)
        CloseHandle(Thread[i]);
    CloseHandle(ThreadP);
    CloseHandle(ThreadV);
    infile.close();
    system("pause");
    return 0;
}