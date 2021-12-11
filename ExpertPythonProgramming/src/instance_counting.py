"""
instance_counting.py

Author : Yusuke Kitamura
Create Date : 2020-04-11 16:57:16
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""


class InstanceCountingClass:
    instances_created = 0

    def __new__(cls, *args, **kwargs):
        print("call __new__(): ", cls, args, kwargs)
        instance = super().__new__(cls)
        instance.number = cls.instances_created
        cls.instances_created += 1
        return instance

    def __init__(self, attribute):
        print("call __init__(): ", self, attribute)
        self.attribute = attribute
