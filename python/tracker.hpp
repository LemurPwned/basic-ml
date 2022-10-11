#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

double computeIOU(const std::vector<double> &box1, const std::vector<double> &box2)
{
    const double x1 = std::max(box1[0], box2[0]);
    const double y1 = std::max(box1[1], box2[1]);
    const double x2 = std::min(box1[2], box2[2]);
    const double y2 = std::min(box1[3], box2[3]);
    const double w = std::max(0.0, x2 - x1 + 1);
    const double h = std::max(0.0, y2 - y1 + 1);
    const double inter = w * h;
    const double o = inter / (box1[2] - box1[0] + 1) / (box1[3] - box1[1] + 1) + inter / (box2[2] - box2[0] + 1) / (box2[3] - box2[1] + 1);
    return o - inter / (box1[2] - box1[0] + 1) / (box1[3] - box1[1] + 1);
}

class Track
{
private:
    std::vector<std::vector<double>> detections;
    double maxScore = 0.0;
    unsigned int shadowCount = 0;

public:
    explicit Track(const std::vector<double> &box)
    {
        detections.push_back(box);
        if (box[4] > maxScore)
            maxScore = box[4];
    }

    std::vector<double> getLastDetection() const
    {
        return detections.back();
    }

    unsigned int getShadowCount() const
    {
        return shadowCount;
    }

    void increaseShadowCount()
    {
        shadowCount++;
    }

    void addDetection(const std::vector<double> &box)
    {
        detections.push_back(box);
    }

    unsigned int getTrackLength() const
    {
        return detections.size();
    }

    double getBestTrackScore() const
    {
        return maxScore;
    }
};

class IOUTracker
{
private:
    unsigned int maxShadowCount;
    unsigned int minTrackLength;
    double iouThreshold;
    double minConfidenceThreshold;
    std::vector<Track> activeTracks;
    std::vector<Track> finishedTracks;

public:
    explicit IOUTracker(unsigned int maxShadowCount,
                        unsigned int minTrackLength,
                        double iouThreshold,
                        double minConfidenceThreshold) : maxShadowCount(maxShadowCount),
                                                         minTrackLength(minTrackLength),
                                                         iouThreshold(iouThreshold),
                                                         minConfidenceThreshold(minConfidenceThreshold) {}

    // pass by copy here
    void update(std::vector<std::vector<double>> detections)
    {
        std::vector<unsigned int> toErase;
        for (size_t t = 0; t < activeTracks.size(); t++)
        {
            auto target = activeTracks[t];
            double bestIOU = 0.0;
            std::vector<double> bestBox;
            // iterator
            int bestIndex = -1;
            for (size_t i = 0; i < detections.size(); i++)
            {
                const double iou = computeIOU(target.getLastDetection(), detections[i]);
                if (bestIOU < iou)
                {
                    bestIOU = iou;
                    // save iterator
                    bestIndex = (int)i;
                    bestBox = detections[i];
                }
            }
            if ((bestIndex >= 0) && (bestIOU >= iouThreshold))
            {
                // add detection to track
                target.addDetection(bestBox);
                // remove detection from detections
                detections.erase(detections.begin() + bestIndex);
            }
            else
            {
                target.increaseShadowCount();
                if ((target.getBestTrackScore() >= minConfidenceThreshold) || (target.getShadowCount() >= maxShadowCount))
                {
                    finishedTracks.push_back(target);
                }
                toErase.push_back((int)t);
            }
        }
        // remove cutoff tracks
        for (const auto &target_i : toErase)
        {
            activeTracks.erase(activeTracks.begin() + target_i);
        }
        // push back remaining detections as new tracks
        std::transform(detections.begin(),
                       detections.end(), std::back_inserter(activeTracks),
                       [](const auto &detection)
                       { return Track(detection); });
    }
    
    std::vector<Track> getFinalTracks()
    {
        std::copy_if(activeTracks.begin(), activeTracks.end(), std::back_inserter(finishedTracks), [this](const auto &track)
                     { return track.getTrackLength() >= minTrackLength; });
        return finishedTracks;
    }
};

#endif // TRACKER_HPP