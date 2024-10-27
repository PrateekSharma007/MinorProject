// backend/server.js
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const PORT = 8080;

const wss = new WebSocket.Server({ port: PORT }, () => {
    console.log(`WebSocket server is running on ws://localhost:${PORT}`);
});

let chunkCounter = 0; // Counter to keep track of each chunk

// Handle incoming WebSocket connections
// backend/server.js

wss.on('connection', (ws) => {
    console.log('Client connected');

    // Listen for incoming audio chunks
    ws.on('message', (message) => {
        // message is an ArrayBuffer or Blob containing the audio data chunk
        console.log('Received audio chunk of size:', message.byteLength);
        
        // Example: Save or process the audio chunk
        // For instance, you could pipe it to an STT model or save it temporarily
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

