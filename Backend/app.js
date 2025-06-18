const http = require("http");
const fs = require("fs").promises;
const path = require("path");
const mongoose = require("mongoose");
const querystring = require("querystring");
const crypto = require("crypto");
const axios = require("axios");
const { spawn } = require("child_process");

let pythonExecuted = false;

// Connect to MongoDB
mongoose.connect("mongodb://localhost/db")
  .then(() => console.log("Connected to MongoDB"))
  .catch(err => console.error("MongoDB connection error:", err));

// User schema and model
const userSchema = new mongoose.Schema({
  username: String,
  password: String,
  sessionToken: String,
  emotionCounts: {
    anger: { type: Number, default: 0 },
    fear: { type: Number, default: 0 },
    joy: { type: Number, default: 0 },
    love: { type: Number, default: 0 },
    sadness: { type: Number, default: 0 },
    surprise: { type: Number, default: 0 }
  }
});
const User = mongoose.model("Users", userSchema, "Users");

// Helper functions
function parseCookies(request) {
  return request.headers.cookie?.split(';').reduce((cookies, pair) => {
    const [name, value] = pair.trim().split('=');
    cookies[name] = value;
    return cookies;
  }, {}) || {};
}

function generateToken() {
  return crypto.randomBytes(32).toString('hex');
}

async function validateSession(req) {
  const cookies = parseCookies(req);
  if (!cookies.sessionToken || !cookies.username) return false;
  const user = await User.findOne({
    username: cookies.username,
    sessionToken: cookies.sessionToken
  });
  return !!user;
}

async function updateUserEmotionCounts(username) {
  console.log("Control comes here");
  try {
    const data = await fs.readFile('detected_emotions.txt', 'utf8');
    const lines = data.split('\n').filter(line => line.trim());
    const counts = {};
    lines.forEach(line => {
      const [emotion, count] = line.split(':').map(s => s.trim());
      if (emotion && count) counts[emotion] = parseInt(count) || 0;
    });
    await User.updateOne(
      { username },
      { $set: { emotionCounts: counts } }
    );
    console.log(`Updated emotion counts for user: ${username}`);
  } catch (err) {
    console.error("Error updating emotion counts:", err);
  }
}

function setSecureCookies(res, sessionToken, username) {
  res.setHeader("Set-Cookie", [
    `sessionToken=${sessionToken}; HttpOnly; SameSite=Strict; Path=/`,
    `username=${username}; Path=/`
  ]);
}

function send404(res) {
  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Page not found");
}

function sendError(res, code, message) {
  res.writeHead(code, { "Content-Type": "text/plain" });
  res.end(message);
}

async function checkPythonExists(pythonCommand) {
  return new Promise((resolve) => {
    const checkProcess = spawn(pythonCommand, ["--version"], { shell: false });
    checkProcess.on("error", () => resolve(false));
    checkProcess.on("exit", (code) => resolve(code === 0));
  });
}

const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];
  const method = req.method;

  // Handle chatbot API requests
  if (url === "/get" && method === "GET") {
    try {
      const msg = new URL(req.url, `http://${req.headers.host}`).searchParams.get('msg');
      if (!msg) return send404(res);
      const response = await axios.get(
        `http://localhost:5000/get?msg=${encodeURIComponent(msg)}`,
        { responseType: 'text' }
      );
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(response.data);
    } catch (error) {
      console.error("Error calling chatbot:", error);
      send404(res);
    }
    return;
  }

  // End of chat handler
  if (url === "/end-chat" && method === "POST") {
    const cookies = parseCookies(req);
    const username = cookies.username;
    if (!username) return send404(res);
    await updateUserEmotionCounts(username);
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("Emotion counts updated");
    return;
  }

  // Dashboard: Fetch emotion counts for visualization
  if (url === "/dashboard" && method === "GET") {
    const cookies = parseCookies(req);
    const username = cookies.username;
    if (!username) {
      send404(res);
      return;
    }
    try {
      const user = await User.findOne({ username });
      if (!user) {
        send404(res);
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      console.log(username);
      res.end(JSON.stringify(user.emotionCounts));
    } catch (err) {
      console.error("Error fetching emotion counts:", err);
      send404(res);
    }
    return;
  }

  try {
    // Serve index
    if (url === "/") {
      if (!pythonExecuted) {
        const chatbotPath = path.join(__dirname, 'chatbot.py');
        try {
          await fs.access(chatbotPath, fs.constants.F_OK);
          pythonExecuted = true;
          const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
          const pythonExists = await checkPythonExists(pythonCommand);
          if (!pythonExists) throw new Error("Python not found");
          const pythonProcess = spawn(
            pythonCommand,
            [chatbotPath],
            {
              stdio: ['pipe', 'pipe', 'pipe'],
              detached: true,
              cwd: path.dirname(chatbotPath),
              env: { ...process.env, PYTHONUNBUFFERED: "1" }
            }
          );
          pythonProcess.stdout.on('data', (data) =>
            console.log(`Python stdout: ${data.toString().trim()}`)
          );
          pythonProcess.stderr.on('data', (data) =>
            console.error(`Python stderr: ${data.toString().trim()}`)
          );
          pythonProcess.on('close', (code) =>
            console.log(`Python process exited with code ${code}`)
          );
          pythonProcess.on('error', (err) =>
            console.error(`Python process failed to start: ${err}`)
          );
        } catch (err) {
          console.error("chatbot.py not found or error:", err.message);
        }
      }
      try {
        const html = await fs.readFile(path.join(__dirname, 'new.html'), "utf8");
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(html);
      } catch (err) {
        console.error("Error reading new.html:", err);
        send404(res);
      }
    }
    // Login handler
    else if (url === "/login" && method === "POST") {
      let body = "";
      req.on("data", chunk => body += chunk);
      req.on("end", async () => {
        try {
          const { username, password } = querystring.parse(body);
          const user = await User.findOne({ username, password });
          if (user) {
            const sessionToken = generateToken();
            user.sessionToken = sessionToken;
            await user.save();
            setSecureCookies(res, sessionToken, username);
            res.writeHead(302, { "Location": "/" });
            res.end();
          } else {
            send404(res);
          }
        } catch (err) {
          console.error("Login error:", err);
          send404(res);
        }
      });
    }
    // Register handler
    else if (url === "/register" && method === "POST") {
      let body = "";
      req.on("data", chunk => body += chunk);
      req.on("end", async () => {
        try {
          const { username, password } = querystring.parse(body);
          if (await User.exists({ username })) {
            send404(res);
            return;
          }
          const sessionToken = generateToken();
          const newUser = new User({ username, password, sessionToken });
          await newUser.save();
          setSecureCookies(res, sessionToken, username);
          res.writeHead(302, { "Location": "/" });
          res.end();
        } catch (err) {
          console.error("Register error:", err);
          send404(res);
        }
      });
    }
    // Static files
    else if (url.endsWith(".html") || url.endsWith(".css") || url.endsWith(".js") || url.endsWith(".png")) {
      const safePath = path.resolve(path.join(__dirname, url));
      if (!safePath.startsWith(path.join(__dirname))) {
        send404(res);
        return;
      }
      try {
        const content = await fs.readFile(safePath);
        const contentType =
          url.endsWith(".css") ? "text/css" :
          url.endsWith(".js") ? "application/javascript" :
          url.endsWith(".png") ? "image/png" : "text/html";
        res.writeHead(200, { "Content-Type": contentType });
        res.end(content);
      } catch (error) {
        if (error.code === 'ENOENT') {
          send404(res);
        } else {
          send404(res);
        }
      }
    }
    // Not found
    else {
      send404(res);
    }
  } catch (error) {
    console.error("Server error:", error);
    send404(res);
  }
});

server.listen(7070, () => {
  console.log("Server running on http://localhost:7070");
});
