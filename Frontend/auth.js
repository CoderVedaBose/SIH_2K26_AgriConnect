const AUTH_API = "http://127.0.0.1:8000";

function setMessage(text, type = "") {
    const el = document.getElementById("authMessage");
    if (!el) return;
    el.textContent = text;
    el.className = `auth-message ${type}`;
}

async function authRequest(endpoint, payload) {
    const response = await fetch(`${AUTH_API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    let data = {};
    try { data = await response.json(); } catch {}
    if (!response.ok) throw new Error(data.detail || "Request failed");
    return data;
}

function saveSession(data) {
    localStorage.setItem("agriconnect_user", JSON.stringify(data));
}

function redirectAfterLogin(role) {
    window.location.href = "index.html";
}

document.addEventListener("DOMContentLoaded", () => {
    const farmerRegister = document.getElementById("farmerRegisterForm");
    const farmerLogin = document.getElementById("farmerLoginForm");
    const buyerLogin = document.getElementById("buyerLoginForm");

    const buyerRegister = document.getElementById("buyerRegisterForm");
    if (buyerRegister) {
        buyerRegister.addEventListener("submit", async (e) => {
            e.preventDefault();
            const button = document.getElementById("submitAuthBtn");
            button.disabled = true; button.textContent = "Creating account...";
            try {
                await authRequest("/auth/buyer/register", {
                    name: document.getElementById("name").value.trim(),
                    phone: document.getElementById("phone").value.trim(),
                    password: document.getElementById("password").value
                });
                setMessage("Registration successful. Redirecting to buyer login...", "success");
                setTimeout(() => window.location.href = "buyer-login.html", 800);
            } catch (err) { setMessage(err.message, "error"); }
            finally { button.disabled = false; button.textContent = "Create Buyer Account"; }
        });
    }

    if (farmerRegister) {
        farmerRegister.addEventListener("submit", async (e) => {
            e.preventDefault();
            const button = document.getElementById("submitAuthBtn");
            button.disabled = true; button.textContent = "Creating account...";
            try {
                const data = await authRequest("/auth/farmer/register", {
                    name: document.getElementById("name").value.trim(),
                    phone: document.getElementById("phone").value.trim(),
                    location: document.getElementById("location").value.trim(),
                    produce: document.getElementById("produce").value.trim(),
                    password: document.getElementById("password").value
                });
                setMessage("Registration successful. Redirecting to farmer login...", "success");
                setTimeout(() => window.location.href = "farmer-login.html", 800);
            } catch (err) { setMessage(err.message, "error"); }
            finally { button.disabled = false; button.textContent = "Create Farmer Account"; }
        });
    }

    if (farmerLogin) {
        farmerLogin.addEventListener("submit", async (e) => {
            e.preventDefault();
            const button = document.getElementById("submitAuthBtn");
            button.disabled = true; button.textContent = "Signing in...";
            try {
                const data = await authRequest("/auth/farmer/login", {
                    phone: document.getElementById("phone").value.trim(),
                    password: document.getElementById("password").value
                });
                saveSession(data);
                setMessage("Login successful. Redirecting...", "success");
                setTimeout(() => redirectAfterLogin("farmer"), 500);
            } catch (err) { setMessage(err.message, "error"); }
            finally { button.disabled = false; button.textContent = "Login as Farmer"; }
        });
    }

    if (buyerLogin) {
        buyerLogin.addEventListener("submit", async (e) => {
            e.preventDefault();
            const button = document.getElementById("submitAuthBtn");
            button.disabled = true; button.textContent = "Signing in...";
            try {
                const data = await authRequest("/auth/buyer/login", {
                    phone: document.getElementById("phone").value.trim(),
                    password: document.getElementById("password").value
                });
                saveSession(data);
                setMessage("Login successful. Redirecting...", "success");
                setTimeout(() => redirectAfterLogin("buyer"), 500);
            } catch (err) { setMessage(err.message, "error"); }
            finally { button.disabled = false; button.textContent = "Login as Buyer"; }
        });
    }
});
