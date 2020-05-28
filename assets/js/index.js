const camera = document.getElementById('camera')

const startVideo = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: {
                width: 500,
                height: 500
            },
            audio: false
        })
        .then(stream => {
            camera.srcObject = stream
        })
        .catch(error => {
            console.error(error)
        })
}

const loadLabels = () => {
    const labels = ['Renato Damazio']
    return Promise.all(labels.map(async label => {
        const descriptions = []
        for(let i = 1; i <= 1; i++){
            const img = await faceapi.fetchImage(`/assets/lib/face-api/labels/${label}/${i}.jpeg`)
            const detections = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor()
            descriptions.push(detections.descriptor)

        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions)
    }))
}

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api/models'),
]).then(startVideo)

camera.addEventListener('play', async () => {
    const canvas = faceapi.createCanvasFromMedia(camera)
    canvasSize = {
        width: camera.width,
        height: camera.height
    }
    const labels = await loadLabels()
    faceapi.matchDimensions(canvas, canvasSize)
    document.body.appendChild(canvas)
    setInterval(async () => {
        const detections = await faceapi
        .detectAllFaces(
            camera, new faceapi.TinyFaceDetectorOptions()
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors()
        const resizeDetections = faceapi.resizeResults(detections, canvasSize)
        const faceMatcher = new faceapi.FaceMatcher(labels, 0.6)
        const results = resizeDetections.map(d =>
            faceMatcher.findBestMatch(d.descriptor)
        )

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
        faceapi.draw.drawDetections(canvas, resizeDetections)
        faceapi.draw.drawFaceLandmarks(canvas, resizeDetections)
        faceapi.draw.drawFaceExpressions(canvas, resizeDetections)
        
        resizeDetections.forEach(detection => {
            const { age, gender, genderProbability } = detection
            new faceapi.draw.DrawTextField([
                `${parseInt(age, 10)} years`,
                `${gender} (${parseInt(genderProbability * 100, 10)})`
            ], detection.detection.box.topRight).draw(canvas)
        })

        results.forEach((result, index) => {
            const box = resizeDetections[index].detection.box
            const { label, distance } = result
            new faceapi.draw.DrawTextField([
                `${label} (${parseInt(distance * 100, 10)})`
            ], box.bottomRight).draw(canvas)
        })


    }, 100)
})


