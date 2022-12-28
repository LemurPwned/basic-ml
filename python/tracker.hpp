#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <iostream>

// define detection datatype
typedef std::vector<double> Detection;
/**
 * @brief Compute Intersection Over Union
 * Returns IOU of two bounding boxes.
 * @param box1
 * @param box2
 * @return double
 */
double
computeIOU(const Detection &box1, const Detection &box2)
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

double featureSimilarityManhattan(const std::vector<double> &a, const std::vector<double> &b)
{
    double sum = 0.0;
    for (unsigned int i = 0; i < a.size(); i++)
    {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}
double featureSimilarityCosine(const std::vector<double> &a, const std::vector<double> &b)
{
    double sum = 0.0;
    for (unsigned int i = 0; i < a.size(); i++)
    {
        sum += a[i] * b[i];
    }
    return (sum + 1) / 2;
}

double computeArea(const Detection &box)
{
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
}

/**
 * @brief NMS algorithm
 * Does non-maximum suppression given boxes and scores.
 * @param boxList List of boxes to perform NMS on. Each box is represented as a list of 4 numbers: [x1, y1, x2, y2, ...]
 * @param iouThreshold IOU threshold for merging boxes.
 * @return std::vector<std::vector<double>>
 */
std::vector<Detection> computeNMS(const std::vector<Detection> &boxList, const double iouThreshold)
{
    std::vector<Detection> result;
    for (unsigned int i = 0; i < boxList.size(); i++)
    {
        const auto &box = boxList[i];
        const double a1 = computeArea(box);
        bool discard = false;
        for (unsigned int j = i + 1; j < boxList.size(); j++)
        {
            const auto &box2 = boxList[j];
            if ((computeIOU(box, box2) > iouThreshold) && (a1 < computeArea(box2)))
            {
                discard = true;
            }
        }
        if (!discard)
            result.push_back(box);
    }
    return result;
}

class Track
{
protected:
    std::vector<Detection> detections;
    double maxScore = 0.0;
    unsigned int shadowCount = 0;
    unsigned int trackId;
    inline static unsigned int trackIDCounter = 0;

public:
    static unsigned int getNewTrackID()
    {
        return trackIDCounter++;
    }

    explicit Track()
    {
        this->trackId = getNewTrackID();
    }

    explicit Track(const Detection &box, unsigned int trackId = 0) : trackId(trackId)
    {
        detections.push_back(box);
        if (box[4] > maxScore)
            maxScore = box[4];
    }

    std::vector<Detection> getDetections() const
    {
        return detections;
    }

    Detection getLastDetection() const
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

    void addDetection(const Detection &box)
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

    bool operator==(const Track &other)
    {
        if (other.trackId == this->trackId)
            return true;
        return false;
    }
};

class IOUTracker
{
private:
    std::vector<Track> activeTracks;
    std::vector<Track> finishedTracks;

protected:
    unsigned int maxShadowCount;
    unsigned int minTrackLength;
    double iouThreshold;
    double minConfidenceThreshold;
    unsigned int trackIdCount = 1;
    bool initialised = false;

public:
    explicit IOUTracker(unsigned int maxShadowCount,
                        unsigned int minTrackLength,
                        double iouThreshold,
                        double minConfidenceThreshold) : maxShadowCount(maxShadowCount),
                                                         minTrackLength(minTrackLength),
                                                         iouThreshold(iouThreshold),
                                                         minConfidenceThreshold(minConfidenceThreshold) {}

    void init(const std::vector<Detection> &primeDetections)
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
    std::vector<Track> update(const std::vector<Detection> &primeDetections)
    {
        if (!initialised)
        {
            init(primeDetections);
            initialised = true;
            return this->activeTracks;
        }
        std::vector<Detection> detections = primeDetections;
        std::vector<unsigned int> toErase; // holds indices of active tracks to erase
        for (size_t t = 0; t < activeTracks.size(); t++)
        {
            double bestIOU = 0.0;
            Detection bestBox;
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
        eraseTracks(toErase);
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

    void eraseTracks(std::vector<unsigned int> toEraseIndx)
    {
        // ascending sort
        std::sort(toEraseIndx.begin(), toEraseIndx.end());
        // remove cutoff tracks before adding new ones
        // remove starting from the latest, i.e. largest index of the list!
        for (int k = (int)toEraseIndx.size() - 1; k >= 0; k--)
        {
            activeTracks.erase(activeTracks.begin() + toEraseIndx[k]);
        }
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
        std::copy_if(activeTracks.begin(), activeTracks.end(), std::back_inserter(finishedTracks), [this](const Track &track)
                     { return track.getTrackLength() >= minTrackLength; });
        return finishedTracks;
    }
};

#endif // TRACKER_HPP
