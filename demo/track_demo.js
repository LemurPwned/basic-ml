function xyxyBoxes2trackerBoxes(bboxes) {
    let trackerVec = new Module.DoubleDoubleVector();
    for (let i = 0; i < bboxes.length; i++) {
        // let det = new Module.DoubleVector();
        const det = new Module.DoubleVector();
        console.log(bboxes[i].bbox)
        det.push_back(bboxes[i].bbox[0]);
        det.push_back(bboxes[i].bbox[1]);
        det.push_back(bboxes[i].bbox[2]);
        det.push_back(bboxes[i].bbox[3]);
        det.push_back(1.0); // score
        trackerVec.push_back(det);
        det.delete();
    }
    return trackerVec;
}

async function startStreaming(tracker) {
    const FPS = 10;
    let streaming = true;
    const model = await cocoSsd.load();
    let video = document.getElementById('videoInput');
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);
    async function processVideo() {
        try {
            if (!streaming) {
                // clean and stop.
                src.delete();
                dst.delete();
                return;
            }
            let begin = Date.now();
            // start processing.

            cap.read(src);
            cv.cvtColor(src, dst, cv.COLOR_RGBA2RGB);
            const tensor = tf.tensor(dst.data, [dst.rows, dst.cols, 3], 'int32')
            let predictions = await model.detect(tensor);
            predictions.forEach(prediction => {
                prediction.bbox = xywh2xyxy(prediction.bbox)
            })
            plotDetections(dst, predictions);
            let trackerInput = xyxyBoxes2trackerBoxes(predictions);
            console.log(trackerInput);
            console.log("DONE")
            let tracks = tracker.update(trackerInput);
            cv.imshow('canvasOutput', dst);
            // schedule the next one.
            let delay = 1000 / FPS - (Date.now() - begin);
            setTimeout(processVideo, delay);
        } catch (err) {
            console.log(err);
        }
    };
    setTimeout(processVideo, 0);
}
