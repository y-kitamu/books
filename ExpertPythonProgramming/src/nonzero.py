"""
nonzero.py

Author : Yusuke Kitamura
Create Date : 2020-04-11 17:25:35
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""


class NonZero(int):
    def __new__(cls, value):
        return super().__new__(cls, value) if value != 0 else None

    def __init__(self, skipped_vlaue):
        print("call __init__(): ", self)
        super().__init__()
