// import fetch from "node-fetch";
// import fs from "fs";
// import path from "path";
//
//
// // Number of images to download
// const NUM_IMAGES = 100;
// const OUTPUT_DIR = path.resolve("./dataset");
//
// if (!fs.existsSync(OUTPUT_DIR)) {
//     fs.mkdirSync(OUTPUT_DIR, { recursive: true });
// }
//
// // Function to download an image
// async function downloadImage(url, filename) {
//     const response = await fetch(url);
//     const arrayBuffer = await response.arrayBuffer();
//     fs.writeFileSync(filename, Buffer.from(arrayBuffer));
//     console.log(`Downloaded: ${filename}`);
// }
//
//
// // Main function
// async function downloadImages() {
//     const url = "https://thispersondoesnotexist.com/";
//     for (let i = 1; i <= NUM_IMAGES; i++) {
//         const filename = path.join(OUTPUT_DIR, `person_${i}.jpg`);
//         await downloadImage(url, filename);
//     }
// }
//
// // Run the script
// downloadImages().catch(console.error);


import fetch from "node-fetch";
import fs from "fs";
import path from "path";

// Number of images to download
const NUM_IMAGES = 400;
const OUTPUT_DIR = path.resolve("./dataset");

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Function to download an image
async function downloadImage(url, filename) {
    try {
        const response = await fetch(url);

        // Validate content type to ensure it's an image
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.startsWith("image")) {
            throw new Error(`Unexpected content type: ${contentType}`);
        }

        // Save image
        const arrayBuffer = await response.arrayBuffer();
        fs.writeFileSync(filename, Buffer.from(arrayBuffer));
        console.log(`Downloaded: ${filename}`);
    } catch (error) {
        console.error(`Failed to download image: ${error.message}`);
    }
}

// Function to retry fetching images
async function fetchImages(url, numImages) {
    let downloaded = 300;

    while (downloaded < numImages) {
        const filename = path.join(OUTPUT_DIR, `person_${downloaded + 1}.jpg`);
        await downloadImage(url, filename);

        downloaded++;
        // Optional delay to avoid overwhelming the server
        await new Promise((resolve) => setTimeout(resolve, 1000)); // 1 second delay
    }
}

// Run the script
const IMAGE_URL = "https://thispersondoesnotexist.com";
fetchImages(IMAGE_URL, NUM_IMAGES).catch(console.error);