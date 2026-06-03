/*
 * boids_module.c — CPython glue for boidforge._native.
 *
 * Bridges NumPy SoA buffers to the pure C kernel in boids_kernel.c using the
 * raw CPython C-API and the NumPy C-API (no pybind11). The module exposes a
 * single `step` function that validates the incoming arrays (contiguous,
 * float32, equal length), releases the GIL, and calls bf_step in place.
 *
 * Skeleton stage: argument validation and the kernel are not wired yet; `step`
 * raises NotImplementedError so the import succeeds while behavior is explicit.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "boids.h"

PyDoc_STRVAR(
    step_doc,
    "step(px, py, vx, vy, *, dt, r_sep, r_ali, r_coh, w_sep, w_ali, w_coh,\n"
    "     max_speed, min_speed, max_force, world_w, world_h, boundary) -> None\n"
    "\n"
    "Advance the simulation one timestep, mutating the four contiguous\n"
    "float32 SoA arrays (px, py, vx, vy) in place. Arrays must be 1-D,\n"
    "C-contiguous, float32, and of equal length. The GIL is released around\n"
    "the compute loop.\n"
    "\n"
    "Not yet implemented (skeleton).");

static PyObject *
bf_py_step(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    (void)args;
    (void)kwargs;
    /* Honest skeleton behavior: no fabricated update. */
    PyErr_SetString(PyExc_NotImplementedError,
                    "boidforge._native.step is not implemented yet");
    return NULL;
}

static PyMethodDef bf_methods[] = {
    {"step", (PyCFunction)(void (*)(void))bf_py_step,
     METH_VARARGS | METH_KEYWORDS, step_doc},
    {NULL, NULL, 0, NULL} /* sentinel */
};

PyDoc_STRVAR(
    module_doc,
    "Native CPython kernel for BoidForge (L4 backend).\n"
    "Operates on contiguous float32 Struct-of-Arrays buffers; contains no I/O\n"
    "and no rendering.");

static struct PyModuleDef bf_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    module_doc,
    -1,
    bf_methods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC
PyInit__native(void)
{
    PyObject *m = PyModule_Create(&bf_module);
    if (m == NULL) {
        return NULL;
    }

    /* Initialize the NumPy C-API; required before any array calls. */
    import_array();

    if (PyModule_AddIntConstant(m, "BOUNDARY_WRAP", BF_BOUNDARY_WRAP) < 0 ||
        PyModule_AddIntConstant(m, "BOUNDARY_REFLECT", BF_BOUNDARY_REFLECT) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
