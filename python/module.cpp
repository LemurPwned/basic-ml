#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "tracker.hpp"
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

    py::module tracker_module = m.def_submodule("tracker", "IOU Tracker");
    py::class_<IOUTracker>(tracker_module, "Tracker")
        .def(py::init<unsigned int, unsigned int, double, double>(),
             "maxShadowCount"_a = 30,
             "minTrackLength"_a = 3,
             "iouThreshold"_a = 0.8,
             "minConfidenceThreshold"_a = 0.6)
        .def("init", &IOUTracker::init, "detections"_a.noconvert())
        .def("update", &IOUTracker::update, "detections"_a.noconvert())
        .def("getActiveTracks", &IOUTracker::getActiveTracks)
        .def("getActiveTrackIds", &IOUTracker::getActiveTrackIds)
        .def("getFinalTracks", &IOUTracker::getFinalTracks);
    py::class_<Track>(tracker_module, "Track")
        .def(py::init<const std::vector<double> &>())
        .def("getBestTrackScore", &Track::getBestTrackScore)
        .def("getTrackLength", &Track::getTrackLength)
        .def("getLastDetection", &Track::getLastDetection)
        .def("getShadowCount", &Track::getShadowCount)
        .def("getId", &Track::getId)
        .def("getDetections", &Track::getDetections);

    // Monte Carlo module
    py::module monte_carlo_module = m.def_submodule("mc", "Monte Carlo");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
