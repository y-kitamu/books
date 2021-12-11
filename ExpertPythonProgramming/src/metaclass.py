"""
metaclass.py

Author : Yusuke Kitamura
Create Date : 2020-04-11 18:24:20
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""


class RevealingMeta(type):
    def __new__(mcs, name, bases, namespace):
        print("call Metaclass.__new__(): ", mcs)
        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        print("call Metaclass.__prepare__(): ", mcs)
        return super().__prepare__(name, bases, **kwargs)

    def __init__(cls, name, bases, namespace, **kwargs):
        print("call Metaclass.__init__(): ", cls)
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        print("call Metaclass.__call__(): ", cls)
        return super().__call__(*args, **kwargs)


class RevealingClass(metaclass=RevealingMeta):
    def __new__(cls):
        print(cls, "__new__ called")
        return super().__new__(cls)

    def __init__(self):
        print(self, "__init__ called")
        super().__init__()


class RevealingClass2(metaclass=RevealingMeta):
    def __new__(cls):
        print(cls, "__new__ called")
        return super().__new__(cls)

    def __init__(self):
        print(self, "__init__ called")
        super().__init__()
