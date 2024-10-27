const APP_ID = "70b4e322ddc542f488ca5d2fc18bb6fb"

let uid = sessionStorage.getItem('uid') // each user's unique id
if(!uid){
    uid = String(Math.floor(Math.random() * 10000)) // generating random user id for the user
    sessionStorage.setItem('uid', uid)
}

let token = null;
let client;

let rtmClient;
let channel;

const queryString = window.location.search
const urlParams = new URLSearchParams(queryString)
let roomId = urlParams.get('room') // in next js use the hook usePathname to extract the room id from the URL string 

if(!roomId){
    roomId = 'main'
}

let displayName = sessionStorage.getItem('display_name')
if(!displayName){
    window.location = 'lobby.html'
}

let localTracks = [] // audio and video streams are stored here 
let remoteUsers = {} // key value pairs are stored containing the user UID and audio video tracks , all the users currently joined

let localScreenTracks;
let sharingScreen = false;

let joinRoomInit = async () => {
    rtmClient = await AgoraRTM.createInstance(APP_ID) // real time messaging service provided by agoraRTM
    await rtmClient.login({uid,token}) // logs into agoraRTM system 

    await rtmClient.addOrUpdateLocalUserAttributes({'name':displayName})

    channel = await rtmClient.createChannel(roomId)
    await channel.join()

    channel.on('MemberJoined', handleMemberJoined)
    channel.on('MemberLeft', handleMemberLeft)
    channel.on('ChannelMessage', handleChannelMessage)

    getMembers()
    addBotMessageToDom(`Welcome to the room ${displayName}! 👋`)

    client = AgoraRTC.createClient({mode:'rtc', codec:'vp8'}) // creating the client object , mode: LIVE OR RTC (optimization algorithm ) , codec : ECONDING METHOD THAT THE BROWSERS USES TO ENCODE THE AUDIO/VIDEO TRACKS
    await client.join(APP_ID, roomId, token, uid) // client joins the room through this function 

    client.on('user-published', handleUserPublished) // whenever a user publishes their video/audio stream this method is invoked to add their streams into the channel and display them onto screens , for multiple remote Users.

    client.on('user-left', handleUserLeft)
}

let joinStream = async () => { // get camera and audjio stream and display it into the dom
    document.getElementById('join-btn').style.display = 'none'
    document.getElementsByClassName('stream__actions')[0].style.display = 'flex'

    localTracks = await AgoraRTC.createMicrophoneAndCameraTracks({}, {encoderConfig:{ // asks for access to audio/video permissions ,
        // all of this is stored inside the localTracks
        width:{min:640, ideal:1920, max:1920},
        height:{min:480, ideal:1080, max:1080}
    }})


    let player = `<div class="video__container" id="user-container-${uid}">
                    <div class="video-player" id="user-${uid}"></div>
                 </div>` // this player is displayed onto the DOM , consisting of video of the user with userID uid

    document.getElementById('streams__container').insertAdjacentHTML('beforeend', player) //'beforeend,afterbegin,afterend,beforebegin' are all options for inserting adjacent html 
    document.getElementById(`user-container-${uid}`).addEventListener('click', expandVideoFrame)

    localTracks[1].play(`user-${uid}`) // audio tracks are stored in index 0 and video tracks are stored in index 1 , we are getting video tracks here , it creates a video tag for us and to append that video tag inside a place , we use .play(provide the id where we want to append the video tag and we are done, the live video tracks are displayed inside the video tag which are appended inside the videoPlayer ID that we have mentioned)


    await client.publish([localTracks[0], localTracks[1]]) // a user joined , and his tracks are being published , this function triggers the handleUserPublished funciton ultimately 
}

let switchToCamera = async () => {
    // this function is invoked when the user cancels the screen sharing , user's camera video track has to be switched to now
    let player = `<div class="video__container" id="user-container-${uid}">
                    <div class="video-player" id="user-${uid}"></div>
                 </div>`

    displayFrame.insertAdjacentHTML('beforeend', player)


    if(localTracks){
        console.log('THE LOCAL TRACKS ARE : ',localTracks);
        await localTracks[0].setMuted(true) 
        await localTracks[1].setMuted(true)
    }else{
        console.log('LOCAL TRACKS ARE NULL CURRENTLY !!!');
    }
    

    document.getElementById('mic-btn').classList.remove('active')
    document.getElementById('screen-btn').classList.remove('active')

    localTracks[1].play(`user-${uid}`)
    await client.publish([localTracks[1]]) // user's camera is on 
}

let handleUserPublished = async (user, mediaType) => { // to handle what happens when another user publishes a stream , to add that stream into the channel that can be displayed on the screen , for multiple remote users
    remoteUsers[user.uid] = user

    await client.subscribe(user, mediaType)

    let player = document.getElementById(`user-container-${user.uid}`)
    if(player === null){
        player = `<div class="video__container" id="user-container-${user.uid}">
                <div class="video-player" id="user-${user.uid}"></div>
            </div>`

        document.getElementById('streams__container').insertAdjacentHTML('beforeend', player)
        document.getElementById(`user-container-${user.uid}`).addEventListener('click', expandVideoFrame)
   
    }

    if(displayFrame.style.display){
        let videoFrame = document.getElementById(`user-container-${user.uid}`)
        videoFrame.style.height = '100px'
        videoFrame.style.width = '100px'
    }

    if(mediaType === 'video'){
        user.videoTrack.play(`user-${user.uid}`) // return a video tag and plays the video inside the user-container-{user.id} id
    }

    if(mediaType === 'audio'){
        user.audioTrack.play()
    }

}

let handleUserLeft = async (user) => {
    delete remoteUsers[user.uid] // deleting the user from the remote users list 
    let item = document.getElementById(`user-container-${user.uid}`) // removing the user frame from the DOM containing his video track
    if(item){
        item.remove()
    }

    if(userIdInDisplayFrame === `user-container-${user.uid}`){
        displayFrame.style.display = null
        
        let videoFrames = document.getElementsByClassName('video__container')

        for(let i = 0; videoFrames.length > i; i++){
            videoFrames[i].style.height = '300px'
            videoFrames[i].style.width = '300px'
        }
    }
}

let toggleMic = async (e) => {
    let button = e.currentTarget

    if(localTracks[0].muted){
        await localTracks[0].setMuted(false)
        button.classList.add('active')
    }else{
        await localTracks[0].setMuted(true)
        button.classList.remove('active')
    }
}

let toggleCamera = async (e) => {
    console.log('INSIDE THIS TOGGLE CAMERA');
    let button = e.currentTarget

    if(localTracks[1].muted){
        await localTracks[1].setMuted(false)
        button.classList.add('active')
    }else{
        await localTracks[1].setMuted(true)
        button.classList.remove('active')
    }
}

let toggleScreen = async (e) => {
    let screenButton = e.currentTarget
    let cameraButton = document.getElementById('camera-btn')

    if(!sharingScreen){
        sharingScreen = true

        screenButton.classList.add('active')
        cameraButton.classList.remove('active')
        cameraButton.style.display = 'none'

        localScreenTracks = await AgoraRTC.createScreenVideoTrack() // asks us which screen we want to share

        let userContainer = document.getElementById(`user-container-${uid}`);
        if(userContainer){
            userContainer.remove()
        }
         // remove the current video track of the user who is sharing the screen 

        displayFrame.style.display = 'block'

        let player = `<div class="video__container" id="user-container-${uid}">
                <div class="video-player" id="user-${uid}"></div>
            </div>`

        displayFrame.insertAdjacentHTML('beforeend', player)
        document.getElementById(`user-container-${uid}`).addEventListener('click', expandVideoFrame)

        userIdInDisplayFrame = `user-container-${uid}`
        localScreenTracks.play(`user-${uid}`) // play the screen track inside this id

        if(localTracks)
        await client.unpublish([localTracks[1]]) // unpublish the video track of the user sharing the screen

        await client.publish([localScreenTracks]) // publishing the screen tracks

        let videoFrames = document.getElementsByClassName('video__container')
        for(let i = 0; videoFrames.length > i; i++){
            if(videoFrames[i].id != userIdInDisplayFrame){
              videoFrames[i].style.height = '100px'
              videoFrames[i].style.width = '100px'
            }
          }


    }else{
        sharingScreen = false 
        cameraButton.style.display = 'block'
        document.getElementById(`user-container-${uid}`).remove()

        if(localScreenTracks)
        await client.unpublish([localScreenTracks]) // unpublishing the screen tracks and we want camera video tracks to be displayed on the DOM

        switchToCamera()
    }
}

let leaveStream = async (e) => {
    e.preventDefault()

    document.getElementById('join-btn').style.display = 'block'
    document.getElementsByClassName('stream__actions')[0].style.display = 'none'

    for(let i = 0; localTracks.length > i; i++){
        localTracks[i].stop()
        localTracks[i].close()
    }

    if(localTracks)
    await client.unpublish([localTracks[0], localTracks[1]])

    if(localScreenTracks){
        console.log('the localScreenTracks are : ',localScreenTracks);
        await client.unpublish([localScreenTracks])
    }

    document.getElementById(`user-container-${uid}`).remove()

    if(userIdInDisplayFrame === `user-container-${uid}`){
        displayFrame.style.display = null

        for(let i = 0; videoFrames.length > i; i++){
            videoFrames[i].style.height = '300px'
            videoFrames[i].style.width = '300px'
        }
    }

    channel.sendMessage({text:JSON.stringify({'type':'user_left', 'uid':uid})})
}



document.getElementById('camera-btn').addEventListener('click', toggleCamera)
document.getElementById('mic-btn').addEventListener('click', toggleMic)
document.getElementById('screen-btn').addEventListener('click', toggleScreen)
document.getElementById('join-btn').addEventListener('click', joinStream) // button clicked to join the stream and display audio/video streams inside the DOM
document.getElementById('leave-btn').addEventListener('click', leaveStream)


joinRoomInit()



// now we want to send all the audio/video tracks to be summarised , for that we are using websockets for sending the data to our server in chunks in real time.
// the models work on that data and we receive the summarizations to be displayed on the frontend in real time

// room_rtc.js
const ws = new WebSocket('ws://localhost:8080'); // Connect to the WebSocket server

ws.addEventListener('open', () => {
    console.log('WebSocket connection established');
});

ws.addEventListener('error', (error) => {
    console.error('WebSocket error:', error);
});

ws.addEventListener('close', () => {
    console.log('WebSocket connection closed');
});

// Function to send audio data chunks to the WebSocket server
function sendAudioChunk(audioChunk) {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(audioChunk);
    }
}

// Capture audio and send it to the WebSocket server
async function captureAndSendAudio() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    // Triggered when there’s an audio data chunk available
    mediaRecorder.ondataavailable = (event) => {
        const audioChunk = event.data;
        sendAudioChunk(audioChunk);
    };

    mediaRecorder.start(500); // Capture audio in 500ms chunks
}

// Start capturing and sending audio
captureAndSendAudio();
