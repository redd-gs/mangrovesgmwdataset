function setup() {
    return {
        input: [{ bands: ["B02", "B03", "B04", "B08"], units: "REFLECTANCE" }],
        output: { bands: 3 }
    };
}

function evaluatePixel(s) {
    // Apply a simple enhancement technique using NDVI
    let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
    
    // Enhance the true color output based on NDVI
    let r = s.B04; // Red
    let g = s.B03; // Green
    let b = s.B02; // Blue

    // Adjust colors based on NDVI
    r = r * (1 + ndvi * 0.5); // Enhance red channel
    g = g * (1 + ndvi * 0.3); // Enhance green channel
    b = b * (1 - ndvi * 0.2); // Reduce blue channel for vegetation

    // Clip values to ensure they are within valid range
    r = Math.min(Math.max(r, 0), 1);
    g = Math.min(Math.max(g, 0), 1);
    b = Math.min(Math.max(b, 0), 1);

    return [b, g, r]; // Return in the order of [Blue, Green, Red]
}