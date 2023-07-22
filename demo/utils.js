

function plotDetections(dstImg, predictions) {
    predictions.forEach(
        prediction => {
            cv.rectangle(
                dstImg,
                {
                    x: prediction.bbox[0],
                    y: prediction.bbox[1]
                }, {
                x: prediction.bbox[2],
                y: prediction.bbox[3]
            },
                [255, 0, 255, 255]
            )

        }

    )
}

function plotTracks(dstImg, tracks) {
    for (let i = 0; i < tracks.size(); i++) {
        const track = tracks.get(i);
        const bbox = track.getLastDetection();
        const id = track.getId().toString();
        cv.rectangle(
            dstImg,
            {
                x: bbox.get(0),
                y: bbox.get(1)
            }, {
            x: bbox.get(2),
            y: bbox.get(3)
        },
            [0, 255, 0, 255]
        )
        cv.rectangle(
            dstImg,
            {
                x: bbox.get(0),
                y: bbox.get(1) - 20
            }, {
            x: bbox.get(0) + id.length * 10,
            y: bbox.get(1)
        },
            [0, 255, 0, 255],
            -1
        )
        cv.putText(
            dstImg,
            id,
            {
                x: bbox.get(0),
                y: bbox.get(1) - 5
            },
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            [0, 0, 0, 255]
        )
    }
}


function xywh2xyxy(bbox) {
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
}
