#!/usr/bin/python
# -*- coding: UTF-8 -*-

class A(object):
    bar = 1
    def __init__(self,name):
        self.name=name
    def func1(self):
        print('foo')
        #=2

    @classmethod
    def func2(cls,name):
        print('func2')
        print(cls.bar)
        cls(name).func1()  # 调用 foo 方法
    @classmethod
    def func3(cls):
        print(cls.name)

b=A('b')
b.func1()
b.func3()
A.func2('a')  # 不需要实例化