import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBVI-SvE1LLYo_iO5oqcwKz_UohP-fecBA",
  authDomain: "fake-news-detection-78562.firebaseapp.com",
  projectId: "fake-news-detection-78562",
  storageBucket: "fake-news-detection-78562.firebasestorage.app",
  messagingSenderId: "245544944244",
  appId: "1:245544944244:web:fc0ea386603de18da4c721"
};

const app  = initializeApp(firebaseConfig);
export const auth = getAuth(app);