"""
test_c_extension.py

Author : Yusuke Kitamura
Create Date : 2020-04-15 17:57:19
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""

def fibonacci(n):
    if n < 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    fibonacci(10)
