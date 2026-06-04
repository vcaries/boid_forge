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

#include <limits.h>

#include "boids.h"

PyDoc_STRVAR(
    step_doc,
    "step(px, py, vx, vy, *, dt, r_sep, r_ali, r_coh, w_sep, w_ali, w_coh,\n"
    "     max_speed, min_speed, max_force, world_w, world_h, boundary) -> None\n"
    "\n"
    "Advance the simulation one timestep, mutating the four contiguous\n"
    "float32 SoA arrays (px, py, vx, vy) in place. Arrays must be 1-D,\n"
    "C-contiguous, writable float32, and of equal length. The GIL is released\n"
    "around the compute loop. Output is bit-identical to the Python backends.");

/*
 * Validate one SoA argument: a 1-D, C-contiguous, writable float32 ndarray.
 * Returns the array (borrowed pointer cast) on success, or NULL with an
 * exception set. ``name`` is used only for error messages.
 */
static PyArrayObject *
as_soa_array(PyObject *obj, const char *name)
{
    if (!PyArray_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy.ndarray", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_TYPE(arr) != NPY_FLOAT32) {
        PyErr_Format(PyExc_TypeError, "%s must be float32", name);
        return NULL;
    }
    if (PyArray_NDIM(arr) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1-D", name);
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return NULL;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_WRITEABLE)) {
        PyErr_Format(PyExc_ValueError, "%s must be writable", name);
        return NULL;
    }
    return arr;
}

static PyObject *
bf_py_step(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;

    PyObject *px_obj, *py_obj, *vx_obj, *vy_obj;
    bf_params_t p;
    int boundary;

    static char *kwlist[] = {
        "px", "py", "vx", "vy",
        "dt", "r_sep", "r_ali", "r_coh",
        "w_sep", "w_ali", "w_coh",
        "max_speed", "min_speed", "max_force",
        "world_w", "world_h", "boundary",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOO$ffffffffffffi", kwlist,
            &px_obj, &py_obj, &vx_obj, &vy_obj,
            &p.dt, &p.r_sep, &p.r_ali, &p.r_coh,
            &p.w_sep, &p.w_ali, &p.w_coh,
            &p.max_speed, &p.min_speed, &p.max_force,
            &p.world_w, &p.world_h, &boundary)) {
        return NULL;
    }
    p.boundary = boundary;

    PyArrayObject *px = as_soa_array(px_obj, "px");
    PyArrayObject *py = as_soa_array(py_obj, "py");
    PyArrayObject *vx = as_soa_array(vx_obj, "vx");
    PyArrayObject *vy = as_soa_array(vy_obj, "vy");
    if (!px || !py || !vx || !vy) {
        return NULL;
    }

    npy_intp n = PyArray_DIM(px, 0);
    if (PyArray_DIM(py, 0) != n || PyArray_DIM(vx, 0) != n ||
        PyArray_DIM(vy, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "px, py, vx, vy must all have the same length");
        return NULL;
    }
    if (n > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "too many boids for the kernel");
        return NULL;
    }

    float *px_d = (float *)PyArray_DATA(px);
    float *py_d = (float *)PyArray_DATA(py);
    float *vx_d = (float *)PyArray_DATA(vx);
    float *vy_d = (float *)PyArray_DATA(vy);

    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = bf_step((int)n, px_d, py_d, vx_d, vy_d, &p);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
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
