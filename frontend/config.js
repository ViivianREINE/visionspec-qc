// Frontend API base for production deployment.
// Update this URL after Render backend deployment.
const DEFAULT_API_BASE = window.location.hostname.includes("localhost")
  ? window.location.origin
  : "https://YOUR-RENDER-SERVICE.onrender.com";
window.API_BASE = window.API_BASE || DEFAULT_API_BASE;
