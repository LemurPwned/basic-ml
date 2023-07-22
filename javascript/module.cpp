#include <emscripten/bind.h>
#include "../python/tracker.hpp"
#include "../python/byte.hpp"
#include "../python/rpca.hpp"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(basic_ml)
{
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
        .function("getActiveTrackIds", &IOUTracker::getActiveTrackIds);

    class_<Track>("Track")
        .constructor<const Detection &>()
        .function("getBestTrackScore", &Track::getBestTrackScore)
        .function("getTrackLength", &Track::getTrackLength)
        .function("getLastDetection", &Track::getLastDetection)
        .function("getShadowCount", &Track::getShadowCount)
        .function("getId", &Track::getId)
        .function("getDetections", &Track::getDetections);

    class_<ByteTracker>("ByteTracker")
        .constructor<double, double, double>()
        .function("update", &ByteTracker::update);

    class_<ByteTrack, base<Track>>("ByteTrack")
        .constructor<const Detection &>();

    // emscripten requires to register vectors explicitly
    register_vector<double>("DoubleVector");
    register_vector<std::vector<double>>("DoubleDoubleVector");
    register_vector<Track>("TrackVector");
    register_vector<ByteTrack>("ByteTrackVector");
    register_vector<std::vector<Track>>("TrackVector2D");
}
