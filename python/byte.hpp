#ifndef BYTE_HPP
#define BYTE_HPP

#include <vector>
#include <iostream>
#include <functional>
#include <algorithm>
#include <iterator>

#include "kalman.hpp"
#include "tracker.hpp"
#include "Hungarian.hpp"

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
        this->trackId = ByteTrack::getNewTrackID();
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
        this->detections.push_back(detection);
        this->trackLen++;
    }

    unsigned int getTrackLength() const
    {
        return this->trackLen;
    }
};

class ByteTracker : public IOUTracker
{
private:
    std::function<double(const Detection &, const Detection &)> simOneFn = computeIOU;
    std::function<double(const Detection &, const Detection &)> simTwoFn = computeIOU;
    std::vector<ByteTrack> activeTracks;
    double highConfidenceThreshold, lowConfidenceThreshold, IOUMatchThreshold;

public:
    explicit ByteTracker(double IOUMatchThreshold = 0.9, double highConfidenceThreshold = 0.5,
                         double lowConfidenceThreshold = 0.1) : IOUTracker(lowConfidenceThreshold),
                                                                highConfidenceThreshold(highConfidenceThreshold),
                                                                lowConfidenceThreshold(lowConfidenceThreshold),
                                                                IOUMatchThreshold(IOUMatchThreshold) {}
    void scoreBasedAssignment(const std::vector<Detection> &boxes,
                              std::vector<Detection> &lowDets,
                              std::vector<Detection> &highDets)
    {
        for (const auto &det : boxes)
        {
            if (det[4] >= this->highConfidenceThreshold)
                highDets.push_back(det);
            else if (det[4] > this->lowConfidenceThreshold)
                lowDets.push_back(det);
        }
    }

    std::pair<std::vector<ByteTrack>, std::vector<Detection>> associateTracks(std::vector<ByteTrack> &predTracks,
                                                                              const std::vector<Detection> &dets,
                                                                              const std::function<double(const Detection &,
                                                                                                         const Detection &)>
                                                                                  simFn = computeIOU)
    {
        // compute the cost matrix
        std::vector<std::vector<double>> costMatrix(predTracks.size(), std::vector<double>(dets.size()));
        for (size_t i = 0; i < predTracks.size(); i++)
        {
            for (size_t j = 0; j < dets.size(); j++)
                costMatrix[i][j] = (1. - simFn(predTracks[i].getLastDetection(), dets[j]));
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
            if ((assignment[assignMentIndx] != -1) && costMatrix[assignMentIndx][assignment[assignMentIndx]] >= this->IOUMatchThreshold) // -1 is no assignment
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

    std::vector<ByteTrack> update(const std::vector<Detection> &primeDetections)
    {
        std::vector<Detection> lowDets = {}, highDets = {};
        scoreBasedAssignment(primeDetections, lowDets, highDets);
        kalmanPredict();
        if (this->activeTracks.size() && highDets.size())
        {
            auto returnPairAssoc1 = associateTracks(
                this->activeTracks,
                highDets,
                this->simOneFn);
            if (returnPairAssoc1.first.size() && lowDets.size())
            {
                auto returnPairAssoc2 = associateTracks(
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
                this->eraseTracks(toErase);
                // create new tracks from the unmatched detections
                for (const auto &det : returnPairAssoc2.second)
                    this->activeTracks.push_back(ByteTrack(det));
            }
        }
        else
        {
            // drop low confidence detections if there are no active tracks
            // create new tracks from the unmatched detections
            for (const auto &det : highDets)
                this->activeTracks.push_back(ByteTrack(det));
        }
        return this->activeTracks;
    };

    void eraseTracks(std::vector<unsigned int> &toEraseIndx)
    {
        // ascending sort
        std::sort(toEraseIndx.begin(), toEraseIndx.end());
        // remove cutoff tracks before adding new ones
        // remove starting from the latest, i.e. largest index of the list!
        for (int k = (int)toEraseIndx.size() - 1; k >= 0; k--)
        {
            this->activeTracks.erase(this->activeTracks.begin() + toEraseIndx[k]);
        }
    }

    void kalmanPredict()
    {
        for (ByteTrack track : this->activeTracks)
        {
            track.predict();
        }
    }

    std::vector<ByteTrack> getActiveTracks()
    {
        return this->activeTracks;
    }
};

#endif // BYTE_HPP
