// import * as tf from '@tensorflow/tfjs';
class Tuple {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    getX() {
        return this.x;
    }

    getY() {
        return this.y;
    }

    add(tupleToAdd) {
        return new Tuple(
            this.x + tupleToAdd.x, this.y + tupleToAdd.y
        );
    }
}

const canvas = document.getElementById("canvas");
canvas.width = 364;
canvas.heigth = 364;

let start_background_color = "white"
let orange = "rgba(254, 133, 1, 0.85)";
let draw_color = "black";
let draw_width = "35";
let is_drawing = false;


let context = canvas.getContext("2d");
context.fillStyle = start_background_color;
context.fillRect(0, 0, canvas.width, canvas.heigth);


canvas.addEventListener("touchstart", start, false);
canvas.addEventListener("touchmove", draw, false);
canvas.addEventListener("mousedown", start, false);
canvas.addEventListener("mousemove", draw, false);

canvas.addEventListener("touchend", stop, false);
canvas.addEventListener("mouseup", stop, false);
canvas.addEventListener("mouseout", stop, false);

let prediction = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08];
let previousPrediction =
    [(-1 / 15), (-1 / 15), (-1 / 15), (-1 / 15), (-1 / 15),
        (-1 / 15), (-1 / 15), (-1 / 15), (-1 / 15), (-1 / 15)];

let maxIndex = -1;
let prevMaxIndex = -1;

drawChart(prediction, previousPrediction, maxIndex, prevMaxIndex);

let listOfMainCoordinates = [
    new Tuple(120, 100),
    new Tuple(280, 97),
    new Tuple(200.5, 215),
    new Tuple(180, 280)
];


function start(event) {
    is_drawing = true;
    context.beginPath();
    context.moveTo(
        event.clientX - canvas.offsetLeft,
        event.clientY - canvas.offsetTop);
    event.preventDefault();
}


function draw(event) {
    if (is_drawing) {
        context.lineTo(
            event.clientX - canvas.offsetLeft,
            event.clientY - canvas.offsetTop);
        context.strokeStyle = draw_color;
        context.lineWidth = draw_width;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.stroke();
    }
    event.preventDefault();
}


function stop(event) {
    if (is_drawing) {
        context.stroke();
        context.closePath();
        is_drawing = false;
    }
    event.preventDefault();
}


function clearCanvas() {
    context.fillStyle = start_background_color;
    context.clearRect(0, 0, canvas.width, canvas.heigth);
    context.fillRect(0, 0, canvas.width, canvas.heigth);

    previousPrediction = prediction;
    prediction = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08];

    prevMaxIndex = maxIndex;
    maxIndex = -1

    drawChart(prediction, previousPrediction, maxIndex, prevMaxIndex);
}


// input the key data, that, when connected by lines, describes a digit
function showExample(listOfMainCoordinates) {
    clearCanvas();

    // create list of little steps to draw digit described by listOfMainCoordinates
    let allCoordinates = [];

    for (let i = 0; i < listOfMainCoordinates.length - 1; i++) {

        let dist = getEuclideanDistance(listOfMainCoordinates[i], listOfMainCoordinates[i + 1]);
        let p1 = listOfMainCoordinates[i];
        let p2 = listOfMainCoordinates[i + 1];
        let amountOfStepsForDist = Math.ceil(dist / 3.0);

        for (let j = 0; j < amountOfStepsForDist; j++) {
            let nextCoord = p1.add(
                new Tuple(
                    (p2.x - p1.x) * (j / amountOfStepsForDist),
                    (p2.y - p1.y) * (j / amountOfStepsForDist)
                ));

            allCoordinates.push(nextCoord);
        }
        allCoordinates.push(p2);
    }

    // Draw digit on canvas.
    context.beginPath();
    context.moveTo(
        allCoordinates[0].x,
        allCoordinates[0].y
    );

    var intervalExampleAnimation = setInterval(showExampleAnimation, 2);
    var i = 0;

    function showExampleAnimation() {
        if (i < allCoordinates.length) {
            // Draw digit.
            context.lineTo(allCoordinates[i].getX(), allCoordinates[i].getY());
            context.strokeStyle = draw_color;
            context.lineWidth = draw_width;
            context.lineCap = "round";
            context.lineJoin = "round";
            context.stroke();

        } else if (i === allCoordinates.length + 40) {
            // Press classify button.
            document.getElementById("save-canvas").style.boxShadow = "0px 0px 3px 3px darkorange";

        } else if (i === allCoordinates.length + 110) {
            // Classify.
            document.getElementById("save-canvas").style.removeProperty("box-shadow");
            classify();
            context.closePath();
            clearInterval(intervalExampleAnimation);
        }
        i = i + 1;
    }
}


function getEuclideanDistance(p1, p2) {
    var deltaX = p2.getX() - p1.getX();
    var deltaY = p2.getY() - p1.getY();
    return Math.sqrt((Math.pow(deltaX, 2) + Math.pow(deltaY, 2)))
}


async function classify() {
    // create model from model_1.json
    const model = await tf.loadLayersModel('models_tfjs/model.json');

    // get and preprocess model input
    const canvas = document.getElementById("canvas");
    let context = canvas.getContext("2d");
    let imgData = context.getImageData(0, 0, canvas.width, canvas.height).data;

    let img2DGreyScale = vectorTo2DGrayScaleImg(imgData, canvas.width, canvas.height);
    let resizedImg = decreaseResolution(img2DGreyScale, canvas.width, canvas.height, 28, 28);

    // preprocess input for model:
    // switch background color from white to black, digit color from black to white
    // and create 3 channels again (rgb), all channels have the same values
    for (let i = 0; i < resizedImg.length; i++) {
        for (let j = 0; j < resizedImg.length; j++) {
            let x = (Math.floor(resizedImg[i][j]) - 255) * (-1);
            resizedImg[i][j] = [x, x, x];
        }
    }

    console.assert(resizedImg.length === 28);
    console.assert(resizedImg[0].length === 28);
    console.assert(resizedImg[0][0].length === 3);

    // predict and get index of max prediction
    previousPrediction = prediction;
    prediction = model.predict(tf.tensor([resizedImg]));

    prevMaxIndex = maxIndex;
    maxIndex = prediction.as1D().argMax().dataSync();

    prediction = prediction.dataSync();

    drawChart(prediction, previousPrediction, maxIndex, prevMaxIndex);
}


function vectorTo2DGrayScaleImg(vector, width, height) {
    // extract one of the three channels of rgb img, since it is a gray scale img, all channels have the same values.
    let grayScaleImg = []
    for (let i = 0; i < width * height; i++) {
        var img = vector[i * 4];
        grayScaleImg.push(img);
    }
    // arrange vector as 2-dimensional array
    let img2D = []
    while (grayScaleImg.length) img2D.push(grayScaleImg.splice(0, width));
    return img2D;
}



function decreaseResolution(inputImg, width, height, newHeight, newWidth) {

    const x = height / newHeight;
    const y = width / newWidth;

    let resizedImg = [];

    for (let i = 0; i < newHeight; i++) {
        let row = [];
        for (let j = 0; j < newWidth; j++) {
            let sum = 0.0;
            for (let k = 0; k < height / newHeight; k++) {
                for (let l = 0; l < width / newWidth; l++) {
                    sum = sum + inputImg[i * x + k][j * y + l];
                }
            }
            row.push(sum / (x * y));
        }
        resizedImg.push(row);
    }
    console.assert(resizedImg.length === 28);
    console.assert(resizedImg[0].length === 28);
    return resizedImg;
}


function drawChart(prediction, previousPrediction, maxIndex, prevMaxIndex) {
    if (prevMaxIndex >= 0) {
        document.getElementById("bar" + prevMaxIndex).classList.remove("shadow");
        document.getElementById("classLabel" + prevMaxIndex).setAttribute("x", "22");
        document.getElementById("classLabel" + prevMaxIndex).setAttribute("y", (prevMaxIndex * 37 + 8));
        document.getElementById("classLabel" + prevMaxIndex).setAttribute("height", "25");
        document.getElementById("classLabel" + prevMaxIndex).setAttribute("width", "25");
        document.getElementById("classLabelText" + prevMaxIndex).setAttribute("x", "28");
    }

    var intervalBarAnimation = setInterval(barAnimation, 0.7);
    var round = 1;
    var stop = 100;

    function barAnimation() {
        for (let i = 0; i < prediction.length; i++) {

            var prevPrediction = previousPrediction[i];
            var currPrediction = prediction[i];

            var minimumBarLength = 7.5;
            var diff = currPrediction - prevPrediction;
            var newWidth = (prevPrediction + (round / stop) * diff) * 300 + minimumBarLength;

            if (newWidth >= 0) {
                document.getElementById("bar" + i).setAttribute("width", newWidth);
            }
        }

        if (round === stop) {
            clearInterval(intervalBarAnimation);
            if (maxIndex >= 0) {
                document.getElementById("bar" + maxIndex).classList.add("shadow");
                document.getElementById("classLabel" + maxIndex).setAttribute("x", "6");
                document.getElementById("classLabel" + maxIndex).setAttribute("y", maxIndex * 37);
                document.getElementById("classLabel" + maxIndex).setAttribute("height", "41");
                document.getElementById("classLabel" + maxIndex).setAttribute("width", "41");
                document.getElementById("classLabelText" + maxIndex).setAttribute("x", "19");
            }
        }
        round = round + 1;
    }
}
