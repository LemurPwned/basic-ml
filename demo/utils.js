

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

function xywh2xyxy(bbox) {
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
}
