#ifndef BYTE_HPP
#define BYTE_HPP

#include <vector>
#include <iostream>
#include <functional>
#include <algorithm>
#include <iterator>

#include "kalman.hpp"
#include "tracker.hpp"
#include "third_party/hungarian-algorithm-cpp/Hungarian.h"

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

enum TrackState
{
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

class ByteTrack : public Track
{
private:
    KalmanFilterTracker kalmanFilter;
    bool activated = false;
    GaussParams8d params; // mean, covariance
    unsigned int trackLen = 0;

public:
    TrackState state = TrackState::New;

    explicit ByteTrack(const Detection &detection) : Track()
    {
        this->activate(detection);
    }
    void predict()
    {
        auto meanCpy = this->params.first;
        if (this->state != TrackState::Tracked)
        {
            meanCpy[7] = 0;
        }
        this->params = this->kalmanFilter.predict(meanCpy, this->params.second);
    }
    void activate(const Detection &detection)
    {
        this->trackId = Track::getNewTrackID();
        this->detections.push_back(detection);
    }
    void reActivate(const Detection &detection)
    {
        this->params = this->kalmanFilter.update(this->params.first,
                                                 this->params.second, detection);
        this->state = TrackState::Tracked;
        this->activated = true;
        this->trackLen = 0;
    }
    void update(const Detection &detection)
    {
        this->params = this->kalmanFilter.update(this->params.first,
                                                 this->params.second, detection);
        this->state = TrackState::Tracked;
        this->trackLen++;
    }
    void markLost()
    {
        this->state = TrackState::Lost;
    }
    void markRemoved()
    {
        this->state = TrackState::Removed;
    }
};

class ByteTracker : public IOUTracker
{

private:
    std::function<double(const Detection &, const Detection &)> simOneFn = computeIOU;
    std::function<double(const Detection &, const Detection &)> simTwoFn = computeIOU;
    std::vector<ByteTrack> activeTracks;

public:
    ByteTracker();
    void lowScoreAssignment(const std::vector<Detection> &boxes,
                            std::vector<Detection> &lowDets,
                            std::vector<Detection> &highDets)
    {
        for (const auto &det : boxes)
        {
            if (det[4] < this->minConfidenceThreshold)
                lowDets.push_back(det);
            else
                highDets.push_back(det);
        }
    }

    std::pair<std::vector<ByteTrack>, std::vector<Detection>> associateTracks(std::vector<ByteTrack> &predTracks,
                                                                              const std::vector<Detection> &dets,
                                                                              const std::function<double(const Detection &,
                                                                                                         const Detection &)>
                                                                                  simFn = computeIOU)
    {
        std::vector<std::vector<double>> costMatrix(predTracks.size(), std::vector<double>(dets.size()));
        for (size_t i = 0; i < predTracks.size(); i++)
        {
            for (size_t j = 0; j < dets.size(); j++)
                costMatrix[i][j] = (1 - simFn(predTracks[i].getLastDetection(), dets[j]));
        }
        // solve the assignment problem
        HungarianAlgorithm HungAlgo;
        std::vector<int> assignment;
        HungAlgo.Solve(costMatrix, assignment);
        // update tracks with high dets
        // assignment is for rows, i.e. tracks
        std::vector<ByteTrack> unassignedTracks;
        std::vector<Detection> unassignedDets;

        // fill the list with tracks that are not assigned
        for (size_t assignMentIndx = 0; assignMentIndx < assignment.size(); assignMentIndx++)
        {
            if (assignment[assignMentIndx] != -1) // -1 is no assignment
                predTracks[assignMentIndx].update(dets[assignment[assignMentIndx]]);
            else
                unassignedTracks.push_back(predTracks[assignMentIndx]);
        }
        // fill the list with dets that are not assigned
        for (size_t detIndx = 0; detIndx < dets.size(); detIndx++)
        {
            if (std::find(assignment.begin(), assignment.end(), detIndx) == assignment.end())
                unassignedDets.push_back(dets[detIndx]);
        }
        return std::pair<std::vector<ByteTrack>, std::vector<Detection>>(unassignedTracks, unassignedDets);
    }

    std::vector<ByteTrack> update(const std::vector<std::vector<double>> &primeDetections)
    {
        std::vector<Detection> lowDets, highDets;
        lowScoreAssignment(primeDetections, highDets, lowDets);
        kalmanPredict();
        auto returnPairAssoc1 = associateTracks(
            this->activeTracks,
            highDets,
            this->simOneFn);
        const auto returnPairAssoc2 = associateTracks(
            returnPairAssoc1.first,
            lowDets,
            this->simTwoFn);

        // delete unmatched tracks
        std::vector<unsigned int> toErase;
        for (auto &track : returnPairAssoc2.first)
        {
            const auto it = std::find(this->activeTracks.begin(),
                                      this->activeTracks.end(), track);
            // push INDEX of the track in the activeTracks vector
            if (it != this->activeTracks.end())
            {
                const std::size_t index = std::distance(this->activeTracks.begin(), it);
                toErase.push_back((int)index);
            }
            else
            {
                std::runtime_error("Track not found in activeTracks");
            }
        }
        // update active Tracks
        eraseTracks(toErase);
        // create new tracks from the unmatched detections
        for (const auto &det : returnPairAssoc2.second)
            this->activeTracks.push_back(ByteTrack(det));

        return this->activeTracks;
    };

    void kalmanPredict()
    {
        for (ByteTrack track : this->activeTracks)
        {
            track.predict();
        }
    }
};

#endif // BYTE_HPP
