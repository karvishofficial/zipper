import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { 
  getFirestore, 
  doc, 
  onSnapshot, 
  setDoc,
  collection,
  query,
  orderBy,
  limit
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

// Firebase configuration is injected at runtime via gitignored config.js (see config.example.js)
const firebaseConfig = window.__ZIPPER_FIREBASE_CONFIG__ || {
  apiKey: "",
  authDomain: "",
  projectId: "",
  storageBucket: "",
  messagingSenderId: "",
  appId: ""
};

// Initialize Firebase only when configured
const app = firebaseConfig.apiKey && firebaseConfig.projectId ? initializeApp(firebaseConfig) : null;
const db = app ? getFirestore(app) : null;
if (!db) {
  console.warn("Firebase config missing: copy web/tracking-dashboard/config.example.js to config.js.");
}

// URL parameters
const urlParams = new URLSearchParams(window.location.search);
const sessionId = urlParams.get("session") || "SESSION_LATEST";

// State
let targetCoords = null;
let viewerCoords = null;
let map = null;
let targetMarker = null;
let targetAccuracyCircle = null;
let viewerMarker = null;
let connectingLine = null;
let pathTrailPolyline = null;
const pathHistory = [];

// DOM Elements
const connectionStatus = document.getElementById("connectionStatus");
const mimicBadge = document.getElementById("mimicBadge");
const liveDistanceText = document.getElementById("liveDistanceText");
const btnCenterTarget = document.getElementById("btnCenterTarget");
const btnCenterMe = document.getElementById("btnCenterMe");
const btnToggleTrail = document.getElementById("btnToggleTrail");
const btnGoogleNavigate = document.getElementById("btnGoogleNavigate");

// Telemetry DOM
const valBattery = document.getElementById("valBattery");
const valCharging = document.getElementById("valCharging");
const batteryMeterFill = document.getElementById("batteryMeterFill");
const valSpeed = document.getElementById("valSpeed");
const valHeading = document.getElementById("valHeading");
const valAccuracy = document.getElementById("valAccuracy");
const valAltitude = document.getElementById("valAltitude");
const valNetwork = document.getElementById("valNetwork");
const valIP = document.getElementById("valIP");
const lastPingTime = document.getElementById("lastPingTime");

// Location DOM
const fullAddressText = document.getElementById("fullAddressText");
const valLat = document.getElementById("valLat");
const valLng = document.getElementById("valLng");
const nearbyList = document.getElementById("nearbyList");

// Evidence DOM
const imgFrontMugshot = document.getElementById("imgFrontMugshot");
const frontPhotoPlaceholder = document.getElementById("frontPhotoPlaceholder");
const imgBackSurroundings = document.getElementById("imgBackSurroundings");
const backPhotoPlaceholder = document.getElementById("backPhotoPlaceholder");

// Lightbox DOM
const photoLightbox = document.getElementById("photoLightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxCaption = document.getElementById("lightboxCaption");
const btnCloseLightbox = document.getElementById("btnCloseLightbox");

// Remote buttons
const btnActionRing = document.getElementById("btnActionRing");
const btnActionSnap = document.getElementById("btnActionSnap");
const btnActionBleScan = document.getElementById("btnActionBleScan");
const btnActionDisarm = document.getElementById("btnActionDisarm");

// 1. Initialize Map
function initMap() {
  // Default to central India (Bangalore baseline: 12.9716, 77.5946) until telemetry locks
  map = L.map("radarMap", {
    zoomControl: false,
    attributionControl: false
  }).setView([12.9716, 77.5946], 14);

  // High-contrast Dark Matter CartoDB Tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 20,
    subdomains: "abcd"
  }).addTo(map);

  L.control.zoom({ position: "bottomright" }).addTo(map);

  // Initialize Trail Polyline
  pathTrailPolyline = L.polyline([], {
    color: "#00e5ff",
    weight: 3,
    dashArray: "6, 8",
    opacity: 0.8
  }).addTo(map);

  // Initialize Connecting Distance Line
  connectingLine = L.polyline([], {
    color: "#ff9100",
    weight: 2,
    dashArray: "4, 6",
    opacity: 0.75
  }).addTo(map);
}

// Custom Leaflet DivIcons
function createTargetIcon() {
  return L.divIcon({
    className: "target-radar-marker",
    html: `<div class="target-wave"></div><div class="target-core"></div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });
}

function createViewerIcon() {
  return L.divIcon({
    className: "viewer-marker",
    html: `<div class="viewer-wave"></div><div class="viewer-core"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });
}

// 2. Start Viewer Geolocation (Dual Device View)
function startViewerTracking() {
  if (!("geolocation" in navigator)) {
    liveDistanceText.textContent = "Viewer geolocation unavailable";
    return;
  }

  navigator.geolocation.watchPosition(
    (pos) => {
      viewerCoords = {
        lat: pos.coords.latitude,
        lng: pos.coords.longitude,
        accuracy: pos.coords.accuracy
      };

      if (!viewerMarker) {
        viewerMarker = L.marker([viewerCoords.lat, viewerCoords.lng], {
          icon: createViewerIcon()
        }).addTo(map).bindPopup("<b>Your Current Location (Viewer)</b>");
      } else {
        viewerMarker.setLatLng([viewerCoords.lat, viewerCoords.lng]);
      }

      updateDistanceAndNavigation();
    },
    (err) => {
      console.warn("Viewer geolocation error:", err.message);
      liveDistanceText.textContent = "Allow location for distance radar";
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 }
  );
}

// 3. Haversine Distance Calculation
function calculateDistanceMeters(lat1, lon1, lat2, lon2) {
  const R = 6371e3; // Earth radius in meters
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) * Math.sin(deltaLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

function updateDistanceAndNavigation() {
  if (!targetCoords) return;

  if (viewerCoords) {
    const distMeters = calculateDistanceMeters(
      viewerCoords.lat,
      viewerCoords.lng,
      targetCoords.lat,
      targetCoords.lng
    );

    let distStr = "";
    if (distMeters < 1000) {
      distStr = `${Math.round(distMeters)} m away`;
    } else {
      distStr = `${(distMeters / 1000).toFixed(2)} km away`;
    }

    const walkMins = Math.max(1, Math.round(distMeters / 80));
    liveDistanceText.textContent = `Target is ${distStr} • ~${walkMins} min walk`;

    // Update Connecting Line
    connectingLine.setLatLngs([
      [viewerCoords.lat, viewerCoords.lng],
      [targetCoords.lat, targetCoords.lng]
    ]);

    // Google Maps Navigation Direct Link
    btnGoogleNavigate.href = `https://www.google.com/maps/dir/?api=1&origin=${viewerCoords.lat},${viewerCoords.lng}&destination=${targetCoords.lat},${targetCoords.lng}&travelmode=walking`;
  } else {
    liveDistanceText.textContent = `Target Active • Satellite fix locked`;
    btnGoogleNavigate.href = `https://www.google.com/maps/search/?api=1&query=${targetCoords.lat},${targetCoords.lng}`;
  }
}

// 4. Reverse Geocoding & Nearby Essentials
let lastGeocodedLat = null;
let lastGeocodedLng = null;

async function reverseGeocode(lat, lng) {
  // Avoid re-fetching if moved less than 30 meters
  if (lastGeocodedLat && lastGeocodedLng) {
    if (calculateDistanceMeters(lastGeocodedLat, lastGeocodedLng, lat, lng) < 30) return;
  }
  lastGeocodedLat = lat;
  lastGeocodedLng = lng;

  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`);
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.display_name) {
      fullAddressText.textContent = data.display_name;
    }
  } catch (e) {
    console.warn("Geocoding lookup failed:", e);
  }

  // Fetch Nearby POIs (Police, Hospitals, Transit)
  fetchNearbyPOIs(lat, lng);
}

async function fetchNearbyPOIs(lat, lng) {
  try {
    // Quick bounding box query around 600m
    const delta = 0.005;
    const overpassQuery = `[out:json][timeout:5];(
      node["amenity"="police"](${lat - delta},${lng - delta},${lat + delta},${lng + delta});
      node["amenity"="hospital"](${lat - delta},${lng - delta},${lng + delta},${lat + delta});
      node["highway"="bus_stop"](${lat - delta},${lng - delta},${lat + delta},${lng + delta});
    );out 4;`;

    const res = await fetch("https://overpass-api.de/api/interpreter", {
      method: "POST",
      body: overpassQuery
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.elements && data.elements.length > 0) {
      nearbyList.innerHTML = "";
      data.elements.slice(0, 4).forEach((el) => {
        const name = el.tags.name || el.tags.amenity || "Local Point";
        const type = el.tags.amenity === "police" ? "🚨 Police Station" : el.tags.amenity === "hospital" ? "🏥 Hospital" : "🚏 Transit Stop";
        const dist = Math.round(calculateDistanceMeters(lat, lng, el.lat, el.lon));
        const li = document.createElement("li");
        li.className = "nearby-item";
        li.textContent = `${type}: ${name} (~${dist}m)`;
        nearbyList.appendChild(li);
      });
    } else {
      nearbyList.innerHTML = `<li class="nearby-item">📍 Residential / Commercial zone detected.</li>`;
    }
  } catch (e) {
    nearbyList.innerHTML = `<li class="nearby-item">📍 Local coordinates recorded and timestamped.</li>`;
  }
}

// 5. Connect to Firebase Firestore Real-Time Stream
function connectFirebaseTelemetry() {
  connectionStatus.className = "status-pill status-connecting";
  connectionStatus.innerHTML = `<span class="pulse-dot"></span><span>SEARCHING RADAR...</span>`;

  // Listen to the specific session document or active target
  const sessionDocRef = doc(db, "tracking_sessions", sessionId);

  onSnapshot(sessionDocRef, (docSnap) => {
    if (!docSnap.exists()) {
      connectionStatus.className = "status-pill status-connecting";
      connectionStatus.innerHTML = `<span class="pulse-dot"></span><span>AWAITING PHONE TELEMETRY...</span>`;
      return;
    }

    const data = docSnap.data();
    connectionStatus.className = "status-pill status-online";
    connectionStatus.innerHTML = `<span class="pulse-dot"></span><span>SATELLITE LOCKED</span>`;

    // State / Mimic
    if (data.isStealthMode || data.triggerReason?.includes("MIMIC") || data.state === "MIMIC_ACTIVE") {
      mimicBadge.classList.remove("hidden");
    } else {
      mimicBadge.classList.add("hidden");
    }

    // Coordinates
    if (data.latitude && data.longitude) {
      targetCoords = {
        lat: Number(data.latitude),
        lng: Number(data.longitude),
        accuracy: Number(data.accuracy || 10)
      };

      valLat.textContent = targetCoords.lat.toFixed(6);
      valLng.textContent = targetCoords.lng.toFixed(6);

      // Update Target Pin
      if (!targetMarker) {
        targetMarker = L.marker([targetCoords.lat, targetCoords.lng], {
          icon: createTargetIcon()
        }).addTo(map).bindPopup("<b>🚨 Stolen Phone (Target Active)</b>");
        map.setView([targetCoords.lat, targetCoords.lng], 16);
      } else {
        targetMarker.setLatLng([targetCoords.lat, targetCoords.lng]);
      }

      // Update Accuracy Circle
      if (!targetAccuracyCircle) {
        targetAccuracyCircle = L.circle([targetCoords.lat, targetCoords.lng], {
          radius: targetCoords.accuracy,
          color: "#ff1744",
          fillColor: "#ff1744",
          fillOpacity: 0.12,
          weight: 1
        }).addTo(map);
      } else {
        targetAccuracyCircle.setLatLng([targetCoords.lat, targetCoords.lng]);
        targetAccuracyCircle.setRadius(targetCoords.accuracy);
      }

      // Trail history
      pathHistory.push([targetCoords.lat, targetCoords.lng]);
      pathTrailPolyline.setLatLngs(pathHistory);

      updateDistanceAndNavigation();
      reverseGeocode(targetCoords.lat, targetCoords.lng);
    }

    // Telemetry values
    if (data.batteryLevel !== undefined) {
      valBattery.textContent = `${data.batteryLevel}%`;
      batteryMeterFill.style.width = `${Math.min(100, Math.max(0, data.batteryLevel))}%`;
      if (data.batteryLevel <= 20) {
        batteryMeterFill.style.backgroundColor = "#ff1744";
      } else if (data.batteryLevel <= 40) {
        batteryMeterFill.style.backgroundColor = "#ff9100";
      } else {
        batteryMeterFill.style.backgroundColor = "#00e676";
      }
    }
    if (data.isCharging !== undefined) {
      valCharging.textContent = data.isCharging ? "⚡ CHARGING" : "STANDBY";
    }
    if (data.speed !== undefined) {
      valSpeed.textContent = Number(data.speed).toFixed(1);
    }
    if (data.heading !== undefined) {
      valHeading.textContent = `HEADING: ${Math.round(data.heading)}°`;
    }
    if (data.accuracy !== undefined) {
      valAccuracy.textContent = `~${Math.round(data.accuracy)}`;
    }
    if (data.altitude !== undefined) {
      valAltitude.textContent = `ALT: ${Math.round(data.altitude)} M`;
    }
    if (data.networkType) {
      valNetwork.textContent = data.networkType.toUpperCase();
    }
    if (data.ipAddress) {
      valIP.textContent = data.ipAddress;
    }
    if (data.timestamp) {
      const date = new Date(data.timestamp);
      lastPingTime.textContent = date.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' });
    }

    // Photo evidence
    if (data.frontPhotoBase64) {
      const src = data.frontPhotoBase64.startsWith("data:") ? data.frontPhotoBase64 : `data:image/jpeg;base64,${data.frontPhotoBase64}`;
      imgFrontMugshot.src = src;
      imgFrontMugshot.classList.remove("hidden");
      frontPhotoPlaceholder.classList.add("hidden");
    }
    if (data.backPhotoBase64) {
      const src = data.backPhotoBase64.startsWith("data:") ? data.backPhotoBase64 : `data:image/jpeg;base64,${data.backPhotoBase64}`;
      imgBackSurroundings.src = src;
      imgBackSurroundings.classList.remove("hidden");
      backPhotoPlaceholder.classList.add("hidden");
    }
  }, (err) => {
    console.error("Firebase Firestore Error:", err);
    connectionStatus.className = "status-pill status-connecting";
    connectionStatus.innerHTML = `<span class="pulse-dot"></span><span>CLOUD RECONNECTING...</span>`;
  });
}

// 6. Remote Actions
async function sendRemoteCommand(actionType, promptText) {
  if (promptText && !confirm(promptText)) return;
  try {
    const sessionDocRef = doc(db, "tracking_sessions", sessionId);
    await setDoc(sessionDocRef, {
      pendingRemoteCommand: {
        action: actionType,
        timestamp: Date.now()
      }
    }, { merge: true });
    alert(`✅ Command [${actionType}] dispatched to phone via Firebase Relay!`);
  } catch (e) {
    alert(`❌ Failed to send command: ${e.message}`);
  }
}

btnActionRing.addEventListener("click", () => sendRemoteCommand("RING", "Sound loud intruder alarm at 100% volume on the phone?"));
btnActionSnap.addEventListener("click", () => sendRemoteCommand("SNAP", "Force immediate dual camera snapshot on the phone?"));
btnActionDisarm.addEventListener("click", () => sendRemoteCommand("DISARM", "Disarm stealth mode and restore phone to normal state?"));

// 7. Web Bluetooth Scanner (Scan Nearby BLE Chirps)
btnActionBleScan.addEventListener("click", async () => {
  if (!navigator.bluetooth) {
    alert("⚠️ Web Bluetooth is only supported on Chrome / Edge browsers with Bluetooth turned ON.");
    return;
  }
  try {
    const device = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true
    });
    alert(`📡 BLE Device spotted nearby: ${device.name || "Unknown Beacon"} (ID: ${device.id})`);
  } catch (e) {
    if (e.name !== "NotFoundError") {
      alert(`BLE Scan note: ${e.message}`);
    }
  }
});

// Map View Controls
btnCenterTarget.addEventListener("click", () => {
  if (targetCoords) map.flyTo([targetCoords.lat, targetCoords.lng], 17);
});
btnCenterMe.addEventListener("click", () => {
  if (viewerCoords) map.flyTo([viewerCoords.lat, viewerCoords.lng], 17);
});
btnToggleTrail.addEventListener("click", () => {
  btnToggleTrail.classList.toggle("active");
  if (btnToggleTrail.classList.contains("active")) {
    map.addLayer(pathTrailPolyline);
  } else {
    map.removeLayer(pathTrailPolyline);
  }
});

// Lightbox modal handlers
function openLightbox(src, caption) {
  if (!src) return;
  lightboxImg.src = src;
  lightboxCaption.textContent = caption;
  photoLightbox.classList.remove("hidden");
}
imgFrontMugshot.addEventListener("click", () => openLightbox(imgFrontMugshot.src, "Front Camera: Intruder Face Mugshot"));
imgBackSurroundings.addEventListener("click", () => openLightbox(imgBackSurroundings.src, "Back Camera: Physical Surroundings"));
btnCloseLightbox.addEventListener("click", () => photoLightbox.classList.add("hidden"));
photoLightbox.addEventListener("click", (e) => {
  if (e.target === photoLightbox) photoLightbox.classList.add("hidden");
});

// Startup
window.addEventListener("DOMContentLoaded", () => {
  initMap();
  startViewerTracking();
  connectFirebaseTelemetry();
});
