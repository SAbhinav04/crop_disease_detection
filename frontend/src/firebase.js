import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAe12J7UkK3r08eMOCV10EGLIYMu2PEWds",
  authDomain: "agrivision-web.firebaseapp.com",
  projectId: "agrivision-web",
  storageBucket: "agrivision-web.firebasestorage.app",
  messagingSenderId: "944843126912",
  appId: "1:944843126912:web:472107699319b337cc0e65",
  measurementId: "G-SYHFWB8767"
};

// Initialize Firebase App
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and export it
export const auth = getAuth(app);
