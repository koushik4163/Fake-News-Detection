import { useEffect, useState } from "react";
import axios from "axios";
import { auth } from "./firebase";
import { supabase } from "./supabase";
import {
  browserLocalPersistence,
  browserSessionPersistence,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  sendPasswordResetEmail,
  setPersistence,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";
import "./App.css";

const PROFILE_TABLE = "users";
const PROFILE_COLUMNS = {
  uid: "uid",
  fullName: "full_name",
  email: "email",
};

const formatAuthError = (err) => {
  const code = err?.code || "";
  if (code === "auth/invalid-credential") return "Invalid email or password.";
  if (code === "auth/email-already-in-use") return "This email is already registered.";
  if (code === "auth/weak-password") return "Password should be at least 6 characters.";
  if (code === "auth/invalid-email") return "Please enter a valid email.";
  if (code === "auth/too-many-requests") return "Too many attempts. Try again later.";
  return err?.message || "Authentication failed. Please try again.";
};

const highlightKeywords = (text, keywords) => {
  if (!keywords?.length) return text;
  const escaped = keywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const regex = new RegExp(`\\b(${escaped.join("|")})\\b`, "gi");
  return text.replace(regex, '<mark style="background:#fef08a;color:#000;padding:0 2px;border-radius:3px">$1</mark>');
};

export default function App() {
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [authError, setAuthError] = useState("");
  const [authMsg, setAuthMsg] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser || null);
    });
    return () => unsubscribe();
  }, []);

  const handleAuth = async (event) => {
    event?.preventDefault();
    setAuthError("");
    setAuthMsg("");

    if (!email || !password) {
      setAuthError("Please enter your email and password.");
      return;
    }
    if (authMode === "signup" && !fullName.trim()) {
      setAuthError("Please enter your full name.");
      return;
    }

    try {
      setAuthLoading(true);
      const persistence = remember ? browserLocalPersistence : browserSessionPersistence;
      await setPersistence(auth, persistence);

      if (authMode === "signup") {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        const profile = {
          [PROFILE_COLUMNS.uid]: credential.user.uid,
          [PROFILE_COLUMNS.fullName]: fullName.trim(),
          [PROFILE_COLUMNS.email]: email.trim(),
        };
        const { error: profileError } = await supabase.from(PROFILE_TABLE).insert(profile);
        if (profileError) {
          setAuthError("Account created, but profile save failed. Check Supabase table/fields.");
          return;
        }
        setAuthMsg("Account created! You are now signed in.");
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
    } catch (err) {
      setAuthError(formatAuthError(err));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    setAuthError("");
    setAuthMsg("");
    if (!email) {
      setAuthError("Enter your email to reset your password.");
      return;
    }
    try {
      await sendPasswordResetEmail(auth, email);
      setAuthMsg("Password reset email sent.");
    } catch (err) {
      setAuthError(formatAuthError(err));
    }
  };

  const handleLogout = async () => {
    await signOut(auth);
    setResult(null);
    setText("");
  };

  const analyze = async () => {
    if (!text.trim()) {
      setError("Please enter a news headline or article.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", { text });
      const data = res.data;
      setResult(data);
      await supabase.from("predictions").insert({
        user_id: user.uid,
        text: text.slice(0, 500),
        label: data.label,
        confidence: data.confidence,
        fake_probability: data.fake_probability,
        real_probability: data.real_probability,
        keywords: data.keywords,
      });
    } catch {
      setError("API error. Make sure the backend is running.");
    }
    setLoading(false);
  };


  if (user) {
    return (
      <div className="app-page">
        <div className="app-shell">
          <header className="app-top">
            <div>
              <p className="app-title">Fake News Detector</p>
              <p className="app-subtitle">{user.email}</p>
            </div>
            <button type="button" className="btn-outline" onClick={handleLogout}>
              Logout
            </button>
          </header>

          <section className="panel">
            <label className="field-label">PASTE NEWS HEADLINE OR ARTICLE</label>
            <textarea
              className="text-area"
              rows={6}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g. Scientists discover new vaccine that cures all diseases overnight..."
            />
            {error && <p className="error-text">{error}</p>}
            <button type="button" className="auth-btn" onClick={analyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze News"}
            </button>
          </section>

          {result && (
            <section className="panel">
              <div className="result-header">
                <div>
                  <p className={`result-title ${result.label === "FAKE" ? "danger" : "success"}`}>
                    {result.label === "FAKE" ? "Fake News" : "Real News"}
                  </p>
                  <p className="app-subtitle">Confidence: {result.confidence}%</p>
                </div>
                <div className="result-score">
                  <span>CREDIBILITY SCORE</span>
                  <strong>{result.real_probability}%</strong>
                </div>
              </div>

              {/* Confidence bars */}
              <div className="bar-group">
                {[["Real News", result.real_probability, "green"], ["Fake News", result.fake_probability, "orange"]].map(([label, val, tone]) => (
                  <div key={label} className="bar-row">
                    <div className="bar-label">
                      <span>{label}</span>
                      <span className={`bar-value ${tone}`}>{val}%</span>
                    </div>
                    <div className="bar-track">
                      <div className={`bar-fill ${tone}`} style={{ width: `${val}%` }} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Keywords */}
              {result.keywords?.length > 0 && (
                <div style={{ marginTop: "18px" }}>
                  <p className="field-label" style={{ marginBottom: "8px" }}>KEY TERMS DETECTED</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                    {result.keywords.map((kw, i) => (
                      <span key={i} style={{ background: "#1f2937", border: "1px solid #374151", color: "#fbbf24", padding: "3px 10px", borderRadius: "999px", fontSize: "12px" }}>
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Highlighted text */}
              <div style={{ marginTop: "18px" }}>
                <p className="field-label" style={{ marginBottom: "8px" }}>YOUR TEXT (keywords highlighted)</p>
                <div
                  style={{ background: "#1f2937", borderRadius: "8px", padding: "12px", color: "#9ca3af", fontSize: "13px", lineHeight: "1.7" }}
                  dangerouslySetInnerHTML={{ __html: highlightKeywords(text, result.keywords) }}
                />
              </div>
            </section>
          )}

        </div>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <div className="auth-shell">
        <section className="auth-form">
          <div className="auth-header">
            <span className="auth-pill">Secure Access</span>
            <h1>{authMode === "login" ? "Welcome Back!" : "Create your account"}</h1>
            <p>
              {authMode === "login"
                ? "Log in to explore credibility insights and your saved checks."
                : "Sign up to start validating news with your personal dashboard."}
            </p>
          </div>

          <form className="auth-fields" onSubmit={handleAuth}>
            {authMode === "signup" && (
              <label className="field">
                <span>Full Name</span>
                <input
                  className="auth-input"
                  name="full_name"
                  type="text"
                  placeholder="Enter your full name"
                  autoComplete="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                />
              </label>
            )}

            <label className="field">
              <span>Email Address</span>
              <input
                className="auth-input"
                name="email"
                type="email"
                placeholder="john@domain.com"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>

            <label className="field">
              <span>Password</span>
              <input
                className="auth-input"
                name="password"
                type="password"
                placeholder="********"
                autoComplete={authMode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>

            <div className="auth-row">
              <label className="auth-check">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                />
                Remember my account
              </label>
              <button type="button" className="auth-link" onClick={handleForgotPassword}>
                Forgot password?
              </button>
            </div>

            {authError && <p className="auth-error">{authError}</p>}
            {authMsg && <p className="auth-success">{authMsg}</p>}

            <button type="submit" className="auth-btn" disabled={authLoading}>
              {authLoading
                ? "Please wait..."
                : authMode === "login"
                  ? "Login"
                  : "Create Account"}
            </button>
          </form>

          <div className="auth-switch">
            {authMode === "login" ? (
              <span>
                Don't have an account?
                <button type="button" onClick={() => { setAuthMode("signup"); setAuthError(""); setAuthMsg(""); }}>
                  Register Now
                </button>
              </span>
            ) : (
              <span>
                Already have an account?
                <button type="button" onClick={() => { setAuthMode("login"); setAuthError(""); setAuthMsg(""); }}>
                  Login
                </button>
              </span>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}