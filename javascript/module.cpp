#include <emscripten/bind.h>
#include "../python/tracker.hpp"
#include "../python/rpca.hpp"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(basic_ml)
{
    // emscripten requires to register vectors explicitly
    register_vector<double>("DoubleVector");
    register_vector<std::vector<double>>("DoubleDoubleVector");
    reigster_vector<std::vector<Track>>("TrackVector");
    class_<RPCA>("RPCA")
        .constructor<unsigned int, double>()
        .function("run", &RPCA::run)
        .function("getL", &RPCA::getL)
        .function("getS", &RPCA::getS);

    class_<IOUTracker>("Tracker")
        .constructor<unsigned int, unsigned int, double, double>()
        .function("init", &IOUTracker::init)
        .function("update", &IOUTracker::update)
        .function("getActiveTracks", &IOUTracker::getActiveTracks)
        .function("getActiveTrackIds", &IOUTracker::getActiveTrackIds)
        .function("getFinalTracks", &IOUTracker::getFinalTracks);

    class_<Track>("Track")
        .constructor<const std::vector<double> &>()
        .function("getBestTrackScore", &Track::getBestTrackScore)
        .function("getTrackLength", &Track::getTrackLength)
        .function("getLastDetection", &Track::getLastDetection)
        .function("getShadowCount", &Track::getShadowCount)
        .function("getId", &Track::getId)
        .function("getDetections", &Track::getDetections);
}
