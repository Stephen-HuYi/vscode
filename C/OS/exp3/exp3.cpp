/*
 * @Description: 
 * @Author: HuYi
 * @Date: 2020-06-06 16:18:33
 * @LastEditors: HuYi
 * @LastEditTime: 2020-06-07 20:03:30
 */
#include <iostream>
#include <windows.h>
#include <iomanip>
using namespace std;

//定义全局变量
SYSTEM_INFO sys_info;                            //用于GetSystemInfo函数返回信息
MEMORY_BASIC_INFORMATION buffer_info;            //用于VirtualQuery函数返回信息
DWORD add_thread_allo;                           //线程Allocator的地址
DWORD add_thread_trac;                           //线程Tracter的地址
HANDLE allo = CreateSemaphore(NULL, 0, 1, NULL); //信号量allo，跟踪线程Allocator
HANDLE trac = CreateSemaphore(NULL, 0, 1, NULL); //信号量trac，跟踪线程Tracker
PBYTE add_trac = NULL;                           //查询虚拟内存区域地址
PBYTE add_allo = PBYTE(0x00200000);              //分配虚拟内存区域地址

void printinfo()
{
    GetSystemInfo(&sys_info); //返回关于当前系统的信息并输出
    cout << "-----系统信息-----" << endl;
    cout << "分页大小: " << sys_info.dwPageSize << endl;
    cout << "最大寻址单元: " << sys_info.lpMaximumApplicationAddress << endl;
    cout << "最小寻址单元: " << sys_info.lpMinimumApplicationAddress << endl;
    cout << "处理器掩码: " << sys_info.dwActiveProcessorMask << endl;
    cout << "处理器个数: " << sys_info.dwNumberOfProcessors << endl;
    cout << "处理器类型: " << sys_info.dwProcessorType << endl;
    cout << "虚拟内存空间的粒度: " << sys_info.dwAllocationGranularity << endl;
    cout << "处理器等级: " << sys_info.wProcessorLevel << endl;
    cout << "处理器版本: " << sys_info.wProcessorRevision << endl;
    cout << endl;
}
void Allocate() //模拟内存分配活动
{
    ReleaseSemaphore(trac, 1, NULL);
    Sleep(100);
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //保留内存
    {
        VirtualAlloc(add_allo, 10 * sys_info.dwPageSize, MEM_RESERVE, PAGE_READWRITE);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //提交内存
    {
        VirtualAlloc(add_allo, 6 * sys_info.dwPageSize, MEM_COMMIT, PAGE_READWRITE);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //回收内存
    {
        VirtualFree(add_allo, sys_info.dwPageSize, MEM_DECOMMIT);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //释放内存
    {
        VirtualFree(add_allo, 0, MEM_RELEASE);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //锁定内存
    {
        VirtualLock(add_allo, 3 * sys_info.dwPageSize);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
    if (WaitForSingleObject(allo, INFINITE) == WAIT_OBJECT_0) //解锁内存
    {
        VirtualUnlock(add_allo, 3 * sys_info.dwPageSize);
        ReleaseSemaphore(trac, 1, NULL);
        Sleep(100);
    }
};

void Track() //跟踪线程Allocate的内存行为
{
    while (1)
    {
        if (WaitForSingleObject(trac, INFINITE) == WAIT_OBJECT_0) //查询虚拟内存分配
        {
            DWORD result = VirtualQuery(add_trac, &buffer_info, sizeof(MEMORY_BASIC_INFORMATION));
            //输出虚拟内存信息
            cout << "区域基地址  分配基地址  区域大小     状态\n";
            while (result && int(add_trac) < 0x00400000)
            {
                if (VirtualQuery(add_trac, &buffer_info, sizeof(buffer_info)) != sizeof(buffer_info))
                    break;
                //输出区域基地址、分配基地址、区域大小
                cout << buffer_info.BaseAddress << "    " << buffer_info.AllocationBase << "    " << setw(8) << hex << buffer_info.RegionSize << ' ';
                switch (buffer_info.State) //输出区域状态
                {
                case MEM_COMMIT:
                    cout << "MEM_COMMIT  ";
                    break;
                case MEM_RESERVE:
                    cout << "MEM_RESERVE ";
                    break;
                case MEM_FREE:
                    cout << "MEM_FREE    ";
                    break;
                default:
                    cout << "----------- ";
                    break;
                }
                cout << endl;
                //寻找下一个虚拟内存
                add_trac = ((PBYTE)buffer_info.BaseAddress + buffer_info.RegionSize);
                result = VirtualQuery(add_trac, &buffer_info, sizeof(MEMORY_BASIC_INFORMATION));
            }
            cout << endl;
            add_trac = NULL;                 //内存置为0
            ReleaseSemaphore(allo, 1, NULL); //释放信息量
        }
    }
};

int main()
{
    printinfo();
    //建立线程Allocator、Tracker
    HANDLE Allocator = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Allocate, NULL, 0, &add_thread_allo);
    HANDLE Tracker = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)Track, NULL, 0, &add_thread_trac);
    Sleep(2000);
    //关闭线程Allocator、Tracker
    CloseHandle(Allocator);
    CloseHandle(Tracker);
    return 0;
}