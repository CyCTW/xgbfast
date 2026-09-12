#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstring>
#include "include/xgbfast.hpp"

typedef struct { PyObject_HEAD xgbfast::Model* model; } NativeModel;
static PyObject* failure() {
    try { throw; }
    catch (const std::invalid_argument& e) { PyErr_SetString(PyExc_ValueError, e.what()); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }
    return nullptr;
}
static int initialize(NativeModel* self, PyObject* args, PyObject*) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return -1;
    if (self->model) { PyErr_SetString(PyExc_RuntimeError, "Model already initialized"); return -1; }
    try { self->model = new xgbfast::Model(path); return 0; }
    catch (...) { failure(); return -1; }
}
static void destroy(NativeModel* self) {
    delete self->model;
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(reinterpret_cast<PyObject*>(self));
    Py_DECREF(type);
}
static PyObject* predict(NativeModel* self, PyObject* input) {
    Py_buffer view{};
    if (PyObject_GetBuffer(input, &view, PyBUF_FORMAT | PyBUF_STRIDES) < 0) return nullptr;
    PyObject* result = nullptr;
    if (view.ndim != 1 || view.itemsize != sizeof(float) || !view.format ||
        (std::strcmp(view.format, "f") && std::strcmp(view.format, "@f") && std::strcmp(view.format, "=f")) ||
        !PyBuffer_IsContiguous(&view, 'C')) {
        PyErr_SetString(PyExc_ValueError, "Expected a contiguous native float32 vector");
    } else {
        try {
            if (!self->model) throw std::runtime_error("Uninitialized model");
            // Hold the GIL for this short call: no Python callback, no reentry,
            // no GIL release/reacquire overhead. Buffer remains pinned.
            result = PyFloat_FromDouble(self->model->predict({static_cast<const float*>(view.buf),
                                              static_cast<size_t>(view.len / sizeof(float))}));
        } catch (...) { result = failure(); }
    }
    PyBuffer_Release(&view);
    return result;
}
static PyObject* close_model(NativeModel* self, PyObject*) {
    try { if (self->model) self->model->close(); Py_RETURN_NONE; }
    catch (...) { return failure(); }
}
static PyObject* enter(NativeModel* self, PyObject*) { return Py_NewRef(self); }
static PyObject* count(NativeModel* self, void*) {
    if (!self->model) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized model"); return nullptr; }
    return PyLong_FromSize_t(self->model->num_features());
}
static PyMethodDef methods[] = {
    {"predict", reinterpret_cast<PyCFunction>(predict), METH_O, "Predict one native float32 vector; return an owned float."},
    {"close", reinterpret_cast<PyCFunction>(close_model), METH_NOARGS, nullptr},
    {"__enter__", reinterpret_cast<PyCFunction>(enter), METH_NOARGS, nullptr},
    {"__exit__", reinterpret_cast<PyCFunction>(close_model), METH_VARARGS, nullptr},
    {nullptr}
};
static PyGetSetDef getters[] = {{"num_features", reinterpret_cast<getter>(count), nullptr, nullptr, nullptr}, {nullptr}};
static PyType_Slot slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)},
    {Py_tp_init, reinterpret_cast<void*>(initialize)},
    {Py_tp_dealloc, reinterpret_cast<void*>(destroy)},
    {Py_tp_methods, methods}, {Py_tp_getset, getters}, {0, nullptr}
};
static PyType_Spec spec = {"xgbfast._native.Model", sizeof(NativeModel), 0, Py_TPFLAGS_DEFAULT, slots};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, nullptr};
PyMODINIT_FUNC PyInit__native() {
    PyObject* m = PyModule_Create(&module);
    if (!m) return nullptr;
    PyObject* type = PyType_FromSpec(&spec);
    if (!type || PyModule_AddObject(m, "Model", type) < 0) { Py_XDECREF(type); Py_DECREF(m); return nullptr; }
    return m;
}
