#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <iostream>

double computeIOU(const std::vector<double> &box1, const std::vector<double> &box2)
{
    const double x1 = std::max(box1[0], box2[0]);
    const double y1 = std::max(box1[1], box2[1]);

    const double x2 = std::min(box1[2], box2[2]);
    const double y2 = std::min(box1[3], box2[3]);

    const double w = std::max(0.0, x2 - x1 + 1);
    const double h = std::max(0.0, y2 - y1 + 1);
    const double inter = w * h;
    const double bbox1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
    const double bbox2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
    const double iou = inter / (bbox1Area + bbox2Area - inter);
    return iou;
}

class Track
{
private:
    std::vector<std::vector<double>> detections;
    double maxScore = 0.0;
    unsigned int shadowCount = 0;
    unsigned int trackId;

public:
    explicit Track(const std::vector<double> &box, unsigned int trackId = 0) : trackId(trackId)
    {
        detections.push_back(box);
        if (box[4] > maxScore)
            maxScore = box[4];
    }

    std::vector<std::vector<double>> getDetections() const
    {
        return detections;
    }

    std::vector<double> getLastDetection() const
    {
        if (detections.size() > 0)
            return detections[detections.size() - 1];
        else
            throw std::runtime_error("No detections in track! It should not happen!");
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

    unsigned int getId() const
    {
        return trackId;
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
    unsigned int trackIdCount = 1;

public:
    explicit IOUTracker(unsigned int maxShadowCount,
                        unsigned int minTrackLength,
                        double iouThreshold,
                        double minConfidenceThreshold) : maxShadowCount(maxShadowCount),
                                                         minTrackLength(minTrackLength),
                                                         iouThreshold(iouThreshold),
                                                         minConfidenceThreshold(minConfidenceThreshold) {}

    void init(const std::vector<std::vector<double>> &primeDetections)
    {
        activeTracks.clear();
        finishedTracks.clear();
        for (const auto &det : primeDetections)
        {
            if (det[4] >= minConfidenceThreshold)
            {
                activeTracks.push_back(Track(det, trackIdCount));
                trackIdCount++;
            }
        }
    }

    // pass by copy here
    std::vector<Track> update(const std::vector<std::vector<double>> &primeDetections)
    {

        std::vector<std::vector<double>> detections = primeDetections;
        std::vector<unsigned int> toErase; // holds indices of active tracks to erase
        for (size_t t = 0; t < activeTracks.size(); t++)
        {
            double bestIOU = 0.0;
            std::vector<double> bestBox;
            // iterator
            int bestIndex = -1;
            for (size_t i = 0; i < detections.size(); i++)
            {
                const double iou = computeIOU(activeTracks[t].getLastDetection(), detections[i]);
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
                activeTracks[t].addDetection(bestBox);
                // remove detection from detections
                detections.erase(detections.begin() + bestIndex);
            }
            else
            { // we haven't found a matching box for this track
                // increase shadow count
                activeTracks[t].increaseShadowCount();
                // if the best score was too low or the shadow count is too high, we finish the track
                if ((activeTracks[t].getBestTrackScore() <= minConfidenceThreshold) || (activeTracks[t].getShadowCount() >= maxShadowCount))
                {
                    finishedTracks.push_back(activeTracks[t]);
                    toErase.push_back((int)t);
                }
            }
        }
        // remove cutoff tracks before adding new ones
        // remove starting from the latest!
        for (int k = (int)toErase.size() - 1; k >= 0; k--)
        {
            activeTracks.erase(activeTracks.begin() + toErase[k]);
        }
        // push back remaining detections as new tracks
        for (const auto &det : detections)
        {
            if (det[4] >= minConfidenceThreshold)
            {
                activeTracks.push_back(Track(det, trackIdCount));
                trackIdCount++;
            }
        }
        return this->activeTracks;
    }

    std::vector<unsigned int> getActiveTrackIds() const
    {
        std::vector<unsigned int> trackIds;
        std::transform(activeTracks.begin(), activeTracks.end(), std::back_inserter(trackIds),
                       [](const Track &track)
                       { return track.getId(); });
        return trackIds;
    }

    std::vector<Track> getActiveTracks()
    {
        return activeTracks;
    }

    std::vector<Track> getFinalTracks()
    {
        std::copy_if(activeTracks.begin(), activeTracks.end(), std::back_inserter(finishedTracks), [this](const auto &track)
                     { return track.getTrackLength() >= minTrackLength; });
        return finishedTracks;
    }
};

#endif // TRACKER_HPP
