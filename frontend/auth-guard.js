// auth-guard.js
// Add <script src="auth-guard.js"></script> as the FIRST script in <body>
// on every page EXCEPT login.html itself.
// If the user is not logged in, they are sent back to login.html instantly.

(function () {
  if (sessionStorage.getItem('pm_admin_logged_in') !== 'true') {
    window.location.replace('login.html');
  }
})();

// Logout helper — call logout() from any page to sign out
function logout() {
  sessionStorage.removeItem('pm_admin_logged_in');
  sessionStorage.removeItem('pm_admin_user');
  sessionStorage.removeItem('pm_login_time');
  window.location.replace('login.html');
}