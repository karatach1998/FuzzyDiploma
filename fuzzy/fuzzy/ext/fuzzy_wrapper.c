#include "Python.h"
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

#include "stdio.h"

#include "fuzzy.h"


static PyObject* predict_cpu_wrapper(PyObject* dummy, PyObject* args);
static PyObject* predict_gpu_wrapper(PyObject* dummy, PyObject* args);

static struct PyMethodDef module_methods[] = {
    {"predict_cpu", &predict_cpu_wrapper, METH_VARARGS, 0},
    {"predict_gpu", &predict_gpu_wrapper, METH_VARARGS, 0},
    { 0 }
};

static struct PyModuleDef fuzzy_module = {
    PyModuleDef_HEAD_INIT,
    "fuzzy",
    "",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_fuzzy_ext(void)
{
    import_array();
    return PyModule_Create(&fuzzy_module);
}

#define PRINT_DEBUG \
    { \
        FILE* f = fopen("tmp.txt", "w"); \
        fputs("Hello from fuzzy extension!\n", f); \
        fclose(f); \
    }


static PyObject* predict_cpu_wrapper(PyObject* dummy, PyObject* args)
{
    return NULL;
}

static PyObject* predict_gpu_wrapper(PyObject* dummy, PyObject* args)
{
    PyObject *fsets_obj;
    PyArrayObject *a0_obj, *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "OOOO", &fsets_obj, &a0_obj, &a_obj, &b_obj)) {
        return NULL;
    }

    /* Check dims */
    Py_ssize_t fsets_len = PyList_Size(fsets_obj);
    npy_intp* a0_dims = PyArray_DIMS(a0_obj); // len == 1
    npy_intp* a_dims = PyArray_DIMS(a_obj); // len == 2
    npy_intp* b_dims = PyArray_DIMS(b_obj); // len == 1

    // if (!(a_dims[0] == b_dims[0] && a0_dims[])) return NULL;

    unsigned N = a_dims[1];
    unsigned n = fsets_len - 1; // NOTE(sergey): last item of `fsets` along 0-axis corresponds to B base fuzzy sets.

    unsigned fsets_lens[fsets_len];
    unsigned fsets_dims[fsets_len];
    unsigned i, j, fset_ptr_buf_len = 0, offset = 0;

    for (i = 0; i < fsets_len; ++i) {
        PyArrayObject* fset_obj = (PyArrayObject*) PyList_GET_ITEM(fsets_obj, i);
        npy_intp* dims = PyArray_DIMS(fset_obj); // len == 2
        fset_ptr_buf_len += fsets_lens[i] = dims[0];
        fsets_dims[i] = dims[1];
    }

    const float* fset_ptr_buf[fset_ptr_buf_len];
    const float** fsets_table[fsets_len];

    for (i = 0; i < fsets_len; ++i) {
        PyArrayObject* fset_obj = (PyArrayObject*) PyList_GET_ITEM(fsets_obj, i);
        fsets_table[i] = fset_ptr_buf + offset;
        for (j = 0; j < fsets_lens[i]; ++j) {
            fsets_table[i][j] = PyArray_GETPTR1(fset_obj, j);
        }
        offset += fsets_lens[i];
    }

    const float* a0[n];
    const unsigned char* a[n];
    const unsigned char* b = PyArray_DATA(b_obj);
    float b0[n+1][fsets_dims[n]];

    for (i = 0; i < n; ++i) {
        a0[i] = PyArray_GETPTR1(a0_obj, i);
        a[i] = PyArray_GETPTR1(a_obj, i);
    }

    predict_gpu(fsets_table, fsets_lens, fsets_dims, a0, a, b, b0, N, n);

    npy_intp b0_dims[2] = { 7, fsets_dims[n] };
    PyObject* b0_obj = PyArray_SimpleNew(2, b0_dims, NPY_FLOAT32);

    memcpy(PyArray_DATA(b0_obj), b0, sizeof(float[n+1][fsets_dims[n]]));
    return b0_obj;
}