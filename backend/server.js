// backend/server.js
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const PORT = 8080;

const wss = new WebSocket.Server({ port: PORT }, () => {
    console.log(`WebSocket server is running on ws://localhost:${PORT}`);
});

let chunkCounter = 0; // Counter to keep track of each chunk
const chunkQueue = []; // Queue to store chunks temporarily

// Create directory for audio chunks if it doesn't exist
const audioDir = path.join(__dirname, 'audio_chunks');
if (!fs.existsSync(audioDir)) {
    fs.mkdirSync(audioDir);
}

// Function to save chunks in batches to avoid frequent I/O operations
const saveChunks = () => {
    if (chunkQueue.length > 0) {
        const chunk = chunkQueue.shift(); // Get the first chunk in the queue
        const chunkPath = path.join(audioDir, `chunk_${chunkCounter}.webm`);

        fs.writeFile(chunkPath, Buffer.from(chunk), (err) => {
            if (err) {
                console.error('Error saving audio chunk:', err);
            } else {
                console.log(`Saved audio chunk as ${chunkPath}`);
                chunkCounter++;
            }
            // Recursive call to process the next chunk in the queue
            saveChunks();
        });
    }
};

wss.on('connection', (ws) => {
    console.log('Client connected dfsdfsa');

    ws.on('message', (message) => {
        console.log('Received audio chunk of size:', message.byteLength);
        chunkQueue.push(message); // Add chunk to the queue

        // Start saving if this is the first item in the queue
        if (chunkQueue.length === 1) saveChunks();
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});
