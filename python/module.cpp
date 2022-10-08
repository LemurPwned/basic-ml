#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "rpca.hpp"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(basic_ml, m)
{
    m.doc() = "Library for basic ML in C++ with Python bindings";
    py::module rpca_module = m.def_submodule("rpca", "Robust PCA");
    py::class_<RPCA>(rpca_module, "RPCA")
        .def(py::init<unsigned int, double>(),
             "maxCount"_a = 1000, "thresholdScale"_a = 1e-7)
        .def("run", &RPCA::run, "X"_a.noconvert())
        .def("getL", &RPCA::getL, py::return_value_policy::reference_internal)
        .def("getS", &RPCA::getS, py::return_value_policy::reference_internal);
    // .def("getFrobeniusNorm", &RPCA::getFrobeniusNorm)
    // .def("shrinkMatrix", &RPCA::shrinkMatrix)
    // .def("truncatedSVD", &RPCA::truncatedSVD);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}