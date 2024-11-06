const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const PORT = 8080;

const wss = new WebSocket.Server({ port: PORT }, () => {
    console.log(`WebSocket server is running on ws://localhost:${PORT}`);
});

const audioDir = path.join(__dirname, 'audio_chunks');
if (!fs.existsSync(audioDir)) {
    fs.mkdirSync(audioDir);
}

wss.on('connection', (ws) => {
    console.log('Client connected');

    // Path for a unique file per session, based on timestamp
    const outputPath = path.join(audioDir, `audio_${Date.now()}.webm`);
    const writeStream = fs.createWriteStream(outputPath);

    ws.on('message', (message) => {
        if (typeof message === 'string' && message === 'stop') {
            // When 'stop' is received, end the write stream
            console.log('Stopping recording and closing file');
            writeStream.end();
        } else {
            // Append the audio chunk to the file
            writeStream.write(Buffer.from(message), (err) => {
                if (err) {
                    console.error('Error saving audio chunk:', err);
                } else {
                    console.log(`Appended audio chunk to ${outputPath}`);
                }
            });
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});


