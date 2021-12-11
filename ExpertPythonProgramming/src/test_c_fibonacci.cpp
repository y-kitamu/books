/*
 * test_c_fibonacci.cpp
 *
 * Create Date : 2020-04-15 18:00:08
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */
#include <Python.h>

long long fibonacci(unsigned int n) {
    if (n < 2) {
        return 1;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

static PyObject *fibonacci_py(PyObject* self, PyObject* args) {
    PyObject *result = NULL;
    long n;
    if (PyArg_ParseTuple(args, "l", &n)) {
        result = Py_BuildValue("L", fibonacci((unsigned int)n));
    }
    return result;
}


static char fibonacci_docs[] =
    "fibonacci(n): Return nth Fibonacci sequence number"
    "computed recursively\n";

static PyMethodDef fibonacci_module_methods[] = {
    {"fibonacci", (PyCFunction)fibonacci_py,
     METH_VARARGS, fibonacci_docs},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fibonacci_module_definition = {
    PyModuleDef_HEAD_INIT,
    "fibonacci",
    "Extension module that provides Fibonacci sequence function",
    -1,
    fibonacci_module_methods
};

PyMODINIT_FUNC PyInit_fibonacci(void) {
    Py_Initialize();
    return PyModule_Create(&fibonacci_module_definition);
}
