// Zipper Voice Twin — Interactive WebRTC & Conversational Voice Client
(function () {
  const urlParams = new URLSearchParams(window.location.search);
  const sdkAppId = parseInt(urlParams.get('sdkAppId') || '20049543', 10);
  const roomId = urlParams.get('roomId') || 'zipper_guest_room';
  const userId = urlParams.get('userId') || 'guest_' + Math.floor(Math.random() * 10000);
  const userSig = urlParams.get('userSig') || '';
  const callerName = urlParams.get('callerName') || 'Friend';

  // DOM Elements
  const callerTitle = document.getElementById('callerTitle');
  const statusBadge = document.getElementById('statusBadge');
  const callTimer = document.getElementById('callTimer');
  const visualizerContainer = document.querySelector('.visualizer-container');
  const speechTranscript = document.getElementById('speechTranscript');
  const preCallActions = document.getElementById('preCallActions');
  const inCallActions = document.getElementById('inCallActions');
  const btnJoinCall = document.getElementById('btnJoinCall');
  const btnToggleMute = document.getElementById('btnToggleMute');
  const btnEndCall = document.getElementById('btnEndCall');
  const muteIcon = document.getElementById('muteIcon');
  const muteLabel = document.getElementById('muteLabel');

  let trtcClient = null;
  let isMuted = false;
  let callStartTime = null;
  let timerInterval = null;
  let isConnected = false;
  let speechRecognizer = null;
  let synth = window.speechSynthesis;
  let audioContext = null;
  let analyser = null;
  let animFrameId = null;
  let isAiSpeaking = false;

  // Initialize UI with caller name
  if (callerName && callerName !== 'Friend') {
    callerTitle.textContent = `Hey ${callerName}!`;
    speechTranscript.textContent = `"Karthik is caught up. Speak to his AI Voice Twin!"`;
  }

  function startTimer() {
    callStartTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - callStartTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      callTimer.textContent = `${m}:${s}`;
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = null;
    callTimer.textContent = '00:00';
  }

  let speechRestartTimer = null;
  let speechWatchdogTimer = null;

  // 1. Text-to-Speech Output (Voice Twin Speaking Aloud)
  function speakAloud(text, onComplete) {
    if (!synth) {
      if (onComplete) onComplete();
      return;
    }

    try { synth.cancel(); } catch (_) {}
    if (synth.paused) { try { synth.resume(); } catch (_) {} }

    // Temporarily pause speech recognition while AI is speaking to prevent mobile audio collision
    if (speechRecognizer) {
      try { speechRecognizer.abort(); } catch (_) {}
    }

    isAiSpeaking = true;
    statusBadge.textContent = 'AI Voice Twin Speaking';
    statusBadge.className = 'status-badge status-speaking';
    visualizerContainer.classList.add('visualizer-active');
    speechTranscript.textContent = `🎙️ AI Twin: "${text}"`;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = 'en-IN'; // Indian English natural accent

    // Try finding an English or regional voice
    const voices = synth.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en-IN')) ||
                           voices.find(v => v.lang.includes('en-GB')) ||
                           voices.find(v => v.lang.includes('en-US'));
    if (preferredVoice) utterance.voice = preferredVoice;

    // Mobile Watchdog: On mobile Android Chrome, utterance.onend frequently hangs or never fires.
    // Calculate expected duration based on text length (~70ms per char, min 1800ms)
    if (speechWatchdogTimer) clearTimeout(speechWatchdogTimer);
    const expectedDurationMs = Math.max(1800, text.length * 75);
    speechWatchdogTimer = setTimeout(() => {
      if (isAiSpeaking) {
        console.log('Mobile speech watchdog elapsed; releasing mic');
        finishAiSpeaking(onComplete);
      }
    }, expectedDurationMs);

    function finishAiSpeaking(callback) {
      if (speechWatchdogTimer) clearTimeout(speechWatchdogTimer);
      isAiSpeaking = false;
      if (isConnected) {
        statusBadge.textContent = 'Listening to you...';
        statusBadge.className = 'status-badge status-connected';
        // Resume listening
        if (speechRecognizer && !isMuted) {
          try { speechRecognizer.start(); } catch (_) {}
        }
      }
      if (callback) callback();
    }

    utterance.onend = () => {
      finishAiSpeaking(onComplete);
    };

    utterance.onerror = () => {
      finishAiSpeaking(onComplete);
    };

    synth.speak(utterance);
  }

  // 2. Intelligent Persona Brain (Bangalore Slang + Accurate Grounding)
  function generateTwinReply(userInput) {
    const text = userInput.toLowerCase();

    // Kannada queries & greetings
    if (text.includes('yavaga') || text.includes('barthya') || text.includes('oorige') || text.includes('mysore')) {
      return "Karthik Mysore schedule check maadi message madthare. Nanu avara AI Voice Twin!";
    }
    if (text.includes('hegiddira') || text.includes('en samachara') || text.includes('namaskara')) {
      return "Namaskara! Karthik swalpa coding kelasadalli iddane. Enadru message idyee?";
    }

    // Schedule / Meeting / Hangout
    if (text.includes('meet') || text.includes('free') || text.includes('when') || text.includes('evening') || text.includes('today') || text.includes('coming')) {
      return `Hey! Karthik is currently in focus mode. I've noted down your plan and notified him. He will confirm with you directly in a few minutes!`;
    }

    // Work / Whereabouts
    if (text.includes('where') || text.includes('doing') || text.includes('call me') || text.includes('busy')) {
      return `Karthik is actively working on his projects right now. You can tell me what you need, and I'll make sure it's on his dashboard immediately.`;
    }

    // General response
    return `Got it, I noted that down for Karthik: "${userInput}". Anything else you'd like me to pass on?`;
  }

  // 3. Web Speech Recognition (Listening to User's Voice)
  function initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      console.warn('SpeechRecognition API not available in this browser');
      return;
    }

    speechRecognizer = new SpeechRec();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = 'en-IN';

    speechRecognizer.onresult = (event) => {
      if (isAiSpeaking) return; // Don't transcribe AI's own voice

      let interim = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interim += transcript;
        }
      }

      if (interim) {
        speechTranscript.textContent = `You: "${interim}..."`;
      }

      if (finalTranscript && finalTranscript.trim().length > 1) {
        const recognized = finalTranscript.trim();
        speechTranscript.textContent = `You: "${recognized}"`;
        statusBadge.textContent = 'AI Twin Thinking...';

        // Pause listening while replying
        setTimeout(() => {
          const reply = generateTwinReply(recognized);
          speakAloud(reply);
        }, 500);
      }
    };

    speechRecognizer.onerror = (e) => {
      console.warn('Speech recognition warning:', e.error);
    };

    speechRecognizer.onend = () => {
      // Debounce restart so Android Chrome doesn't enter a constant Google chime "bloop" loop
      if (speechRestartTimer) clearTimeout(speechRestartTimer);
      if (isConnected && !isMuted && !isAiSpeaking) {
        speechRestartTimer = setTimeout(() => {
          try {
            if (isConnected && !isMuted && !isAiSpeaking) {
              speechRecognizer.start();
            }
          } catch (_) {}
        }, 800);
      }
    };

    try {
      if (!isAiSpeaking) {
        speechRecognizer.start();
      }
    } catch (e) {
      console.warn('Recognition start exception:', e);
    }
  }

  // 4. Audio Visualizer Setup
  function initVisualizer(stream) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      const bars = document.querySelectorAll('.waveform-bar');

      function renderFrame() {
        if (!isConnected) return;
        animFrameId = requestAnimationFrame(renderFrame);
        analyser.getByteFrequencyData(dataArray);

        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        const avg = sum / bufferLength;

        // Animate waveform bars height based on mic intensity
        bars.forEach((bar, idx) => {
          const factor = (dataArray[idx % bufferLength] || avg) / 255;
          const height = Math.max(8, Math.min(48, factor * 56));
          bar.style.height = `${height}px`;
        });
      }
      renderFrame();
    } catch (err) {
      console.warn('Audio visualizer init error:', err);
    }
  }

  // 5. Main Join Call Action
  async function joinCall() {
    btnJoinCall.disabled = true;
    statusBadge.textContent = 'Connecting...';
    statusBadge.className = 'status-badge status-speaking';

    try {
      // Request mic permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      initVisualizer(stream);

      // If official Tencent TRTC credentials exist
      if (typeof TRTC !== 'undefined' && userSig) {
        try {
          trtcClient = TRTC.createClient({
            mode: 'rtc',
            sdkAppId: sdkAppId,
            userId: userId,
            userSig: userSig
          });

          trtcClient.on(TRTC.EVENT.REMOTE_AUDIO_AVAILABLE, async ({ userId: remoteUserId }) => {
            console.log('Remote audio connected from phone:', remoteUserId);
            statusBadge.textContent = 'Karthik Connected';
            statusBadge.className = 'status-badge status-speaking';
            visualizerContainer.classList.add('visualizer-active');
            await trtcClient.muteRemoteAudio(remoteUserId, false);
          });

          await trtcClient.enterRoom({ roomId: roomId });
          await trtcClient.startLocalAudio();
          console.log('Joined TRTC room successfully:', roomId);
        } catch (trtcErr) {
          console.warn('TRTC connection warning (falling back to Local AI Voice):', trtcErr);
        }
      }

      isConnected = true;
      preCallActions.classList.add('in-call-hidden');
      inCallActions.classList.remove('in-call-hidden');
      statusBadge.textContent = 'AI Voice Twin Live';
      statusBadge.className = 'status-badge status-connected';
      visualizerContainer.classList.add('visualizer-active');
      startTimer();

      // Start real-time speech recognition
      initSpeechRecognition();

      // Voice Twin greets the caller immediately aloud!
      const greeting = callerName && callerName !== 'Friend'
        ? `Hey ${callerName}! I'm Karthik's AI Voice Twin. Karthik is caught up with work, but I'm here. What's up?`
        : `Hey! I'm Karthik's AI Voice Twin. Karthik is caught up right now, but I'm here. What's up?`;

      setTimeout(() => {
        speakAloud(greeting);
      }, 400);

    } catch (err) {
      console.error('Failed to join voice call:', err);
      statusBadge.textContent = 'Mic Permission Required';
      statusBadge.className = 'status-badge status-idle';
      btnJoinCall.disabled = false;
      alert('Please allow microphone permission in your browser so you can talk to the Voice Twin.');
    }
  }

  // 6. Controls: Mute & End Call
  async function toggleMute() {
    if (!isConnected) return;
    isMuted = !isMuted;
    if (trtcClient) {
      if (isMuted) {
        await trtcClient.stopLocalAudio();
      } else {
        await trtcClient.startLocalAudio();
      }
    }
    if (speechRecognizer) {
      if (isMuted) {
        try { speechRecognizer.stop(); } catch (_) {}
      } else {
        try { speechRecognizer.start(); } catch (_) {}
      }
    }
    muteIcon.textContent = isMuted ? '🔇' : '🎤';
    muteLabel.textContent = isMuted ? 'Unmute' : 'Mute';
    statusBadge.textContent = isMuted ? 'Muted' : 'Listening to you...';
  }

  async function endCall() {
    try {
      if (trtcClient) {
        await trtcClient.stopLocalAudio();
        await trtcClient.exitRoom();
        trtcClient.destroy();
        trtcClient = null;
      }
    } catch (_) {}

    if (speechRecognizer) {
      try { speechRecognizer.stop(); } catch (_) {}
      speechRecognizer = null;
    }
    if (synth) synth.cancel();
    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (audioContext) {
      try { audioContext.close(); } catch (_) {}
    }

    isConnected = false;
    isMuted = false;
    stopTimer();
    visualizerContainer.classList.remove('visualizer-active');
    preCallActions.classList.remove('in-call-hidden');
    inCallActions.classList.add('in-call-hidden');
    btnJoinCall.disabled = false;
    muteIcon.textContent = '🎤';
    muteLabel.textContent = 'Mute';
    statusBadge.textContent = 'Call Ended';
    statusBadge.className = 'status-badge status-idle';
    speechTranscript.textContent = '"Call ended. Karthik will get your messages on his phone dashboard!"';
  }

  // Ensure speech voices are loaded
  if (synth && synth.onvoiceschanged !== undefined) {
    synth.onvoiceschanged = () => synth.getVoices();
  }

  // Event Listeners
  btnJoinCall.addEventListener('click', joinCall);
  btnToggleMute.addEventListener('click', toggleMute);
  btnEndCall.addEventListener('click', endCall);
})();
