const fetch = require("node-fetch");
const fs = require("fs");
const path = require("path");

// Number of images to download
const NUM_IMAGES = 10;
const OUTPUT_DIR = "./Users/juliayoo/Desktop/NN/dataset";

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR);
}

// Function to download an image
async function downloadImage(url, filename) {
    const response = await fetch(url);
    const buffer = await response.buffer();
    fs.writeFileSync(filename, buffer);
    console.log(`Downloaded: ${filename}`);
}

// Main function
async function downloadImages() {
    const url = "https://thispersondoesnotexist.com/image";
    for (let i = 1; i <= NUM_IMAGES; i++) {
        const filename = path.join(OUTPUT_DIR, `person_${i}.jpg`);
        await downloadImage(url, filename);
    }
}

// Run the script
downloadImages().catch(console.error);
