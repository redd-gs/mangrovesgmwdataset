function setup() {
    return {
        input: [{bands: ["B02", "B03", "B04"], units: "REFLECTANCE"}],
        output: {bands: 3}
    };
}

function evaluatePixel(s) {
    return [s.B04, s.B03, s.B02];
}