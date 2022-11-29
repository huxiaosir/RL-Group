#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/2 10:58
# @Author  : Joisen
# @File    : test2.py

def solu():
    a,b,n = map(int,input().split())

    res = 0
    list = []
    while(a!=0 and b!=0 and n!=0):
        max = int((n*a*b)/(a*b-a-1))
        list.append(max)
        a,b,n = map(int,input().split())
    for x in list:
        print(x)

solu()

'''
public class MaxBottle {
    List<Long> maxBottle (){
        Scanner scanner=new Scanner(System.in);
        long a=scanner.nextInt();//瓶盖数
        long b=scanner.nextInt();//酒瓶数
        long n=scanner.nextInt();//原本酒瓶数
        long max=0;
        List <Long> list=new ArrayList<>();
        while (a!=0&&b!=0&&n!=0){
            max=(n*a*b)/(a*b-a-1);
            list.add(max);
            a=scanner.nextInt();
            b=scanner.nextInt();
            n=scanner.nextInt();
        }
        return list;
    }

    public static void main(String[] args) {
        MaxBottle max=new MaxBottle();
        List<Long> x=max.maxBottle();
        for(Long k:x){
            System.out.println(k);
        }
    }
}
'''