#include <stdbool.h>
#include <Python.h>

typedef struct {
    PyObject_HEAD;
    char *voc_file;
    char *settings_file;
    int sensor;
    bool use_viewer;
} Orb;

static void
Orb_dealloc(Orb* self)
{
    Py_XDECREF(self->voc_file);
    Py_XDECREF(self->settings_file);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Orb_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Orb *self;

    self = (Orb *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->voc_file = "";
        self->settings_file = "";
        self->sensor = 0;
        self->use_viewer = 0;
    }

    return (PyObject *)self;
}

static int
Orb_init(Orb *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"voc_file", "settings_file", "sensor","use_viewer", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "sssp", kwlist,
                                      &self->voc_file, &self->settings_file,
                                      &self->sensor, &self->use_viewer))
        return -1;

    return 0;
}


static PyMemberDef Orb_members[] = {
    {"voc_file", T_STRING, offsetof(Orb, voc_file), 0,
     "Vocabulary file"},
    {"settings_file", T_STRING, offsetof(Orb, settings_file), 0,
     "Settings file"},
    {"sensor", T_INT, offsetof(Orb, sensor), 0,
     "sensor"},
    {"use_viewer", T_INT, offsetof(Orb, use_viewer), 0,
     "use the viewer"},
    {NULL}  /* Sentinel */
};

static PyObject *
Orb_test(Orb* self)
{
    printf("Test\n");
}

static PyMethodDef Orb_methods[] = {
    {"test", (PyCFunction)Orb_test, METH_NOARGS,
     "Test"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject OrbType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "orb.Orb",             /* tp_name */
    sizeof(Orb),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Orb_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Orb objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Orb_methods,             /* tp_methods */
    Orb_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Orb_init,      /* tp_init */
    0,                         /* tp_alloc */
    Orb_new,                 /* tp_new */
};

static PyModuleDef orbmodule = {
    PyModuleDef_HEAD_INIT,
    "orb",
    "Orb slam module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_orb(void)
{
    PyObject* m;

    if (PyType_Ready(&OrbType) < 0)
        return NULL;

    m = PyModule_Create(&orbmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&OrbType);
    PyModule_AddObject(m, "Orb", (PyObject *)&OrbType);
    return m;
}
