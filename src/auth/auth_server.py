import os
import threading
import webbrowser
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.serving import make_server

from src.auth.authenticator import KiteAuthenticator
from src.auth.token_manager import TokenManager
from src.utils.config_loader import AuthServerConfig, BrokerConfig
from src.utils.logger import LOGGER as logger

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
auth_success_event = threading.Event()
auth_error_event = threading.Event()
auth_result = {"success": False, "message": ""}

# Global instances for authenticator and token manager
# These will be set by the start_auth_server_and_wait function
global_token_manager_instance: Optional[TokenManager] = None
global_authenticator_instance: Optional[KiteAuthenticator] = None


class FlaskServer(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()
        self.stopped = threading.Event()

    def run(self) -> None:
        logger.info(f"Starting Flask server on http://{self.host}:{self.port}")
        try:
            self.srv.serve_forever()
        finally:
            self.stopped.set()

    def shutdown(self) -> None:
        logger.info("Shutting down Flask server.")
        self.srv.shutdown()
        if self.srv.socket:
            self.srv.socket.close()
        self.stopped.set()


@app.route("/")
def auth_home() -> str:
    """Main authentication page with token status and login options."""
    return render_template("auth_home.html")


@app.route("/auth/status")
def auth_status() -> Response:
    """API endpoint to check current token status."""
    global global_token_manager_instance

    if global_token_manager_instance is None:
        return jsonify({"valid": False, "error": "Token manager not initialized"})

    try:
        token = global_token_manager_instance.get_access_token()
        if token:
            # Create a preview of the token (first 10 chars + ...)
            token_preview = f"{token[:10]}..." if len(token) > 10 else token
            return jsonify(
                {
                    "valid": True,
                    "token_preview": token_preview,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        return jsonify({"valid": False, "error": "No access token found"})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/auth/login")
def auth_login() -> str:
    """Initiate the Zerodha OAuth flow."""
    global global_authenticator_instance

    if global_authenticator_instance is None:
        return render_template(
            "auth_error.html", error_type="Configuration Error", error_message="Authenticator not initialized"
        )

    try:
        auth_url = global_authenticator_instance.generate_login_url()
        logger.info(f"Redirecting to Zerodha OAuth: {auth_url}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Redirecting to Zerodha...</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .container {{
                    text-align: center;
                    background: rgba(255,255,255,0.1);
                    padding: 40px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                }}
                .spinner {{
                    border: 3px solid rgba(255,255,255,0.3);
                    border-top: 3px solid white;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            <script>
                setTimeout(function() {{
                    window.location.href = "{auth_url}";
                }}, 2000);
            </script>
        </head>
        <body>
            <div class="container">
                <h1>🔐 Connecting to Zerodha</h1>
                <div class="spinner"></div>
                <p>You will be redirected to Zerodha's secure login page...</p>
                <p><small>If you are not redirected automatically, <a href="{auth_url}" style="color: #ffeb3b;">click here</a></small></p>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Error generating login URL: {e}")
        return render_template("auth_error.html", error_type="Login Error", error_message=str(e))


@app.route("/auth/refresh")
def auth_refresh() -> str:
    """Refresh token by reinitializing the OAuth flow."""
    return auth_login()


@app.route("/callback")
def auth_callback() -> str:
    global auth_result
    global global_authenticator_instance
    global global_token_manager_instance
    request_token = request.args.get("request_token")
    status = request.args.get("status")
    error = request.args.get("error")
    error_message = request.args.get("message")

    if status == "success" and request_token:
        try:
            # Use the global authenticator instance
            if global_authenticator_instance is None:
                raise RuntimeError("Authenticator instance not initialized")

            access_token = global_authenticator_instance.authenticate(request_token)
            auth_result["success"] = True
            auth_result["message"] = "Authentication successful! You can close this window."
            logger.info("Authentication flow completed successfully.")
            auth_success_event.set()

            # Create token preview for display
            token_preview = f"{access_token[:10]}..." if len(access_token) > 10 else access_token
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return render_template("auth_success.html", token_preview=token_preview, timestamp=timestamp)
        except Exception as e:
            auth_result["success"] = False
            auth_result["message"] = f"Authentication failed: {e}"
            logger.error(f"Authentication failed during token exchange: {e}")
            auth_error_event.set()

            return render_template(
                "auth_error.html", error_type="Token Exchange Error", error_message=str(e), request_token=request_token
            )
    else:
        auth_result["success"] = False
        auth_result["message"] = f"Authentication failed: {error} - {error_message}"
        logger.error(f"Authentication failed from Kite: {error} - {error_message}")
        auth_error_event.set()

        return render_template(
            "auth_error.html",
            error_type=error or "OAuth Error",
            error_message=error_message or "Unknown authentication error",
            request_token=request_token,
        )


def start_auth_server_and_wait(broker_config: BrokerConfig, auth_server_config: AuthServerConfig) -> bool:
    global global_token_manager_instance, global_authenticator_instance

    # Initialize the global instances
    global_token_manager_instance = TokenManager()
    global_authenticator_instance = KiteAuthenticator(broker_config)

    if not broker_config.redirect_url:
        raise ValueError("Broker redirect URL is not configured.")

    host: Optional[str] = urlparse(broker_config.redirect_url).hostname
    initial_port: Optional[int] = urlparse(broker_config.redirect_url).port

    if host is None or initial_port is None:
        raise ValueError(f"Invalid redirect URL configured: {broker_config.redirect_url}. Host or port is missing.")

    server: FlaskServer
    current_port = initial_port
    max_retries = auth_server_config.max_retries
    # Use 0.0.0.0 to bind to all interfaces in container, but keep the original host for redirect URL
    server_host = auth_server_config.server_host
    for attempt in range(max_retries):
        try:
            server = FlaskServer(server_host, current_port)
            server.daemon = True  # Allow the main program to exit even if the server is running
            server.start()
            # Get the actual port if 0 was used
            if current_port == 0:
                current_port = server.srv.socket.getsockname()[1]
                logger.info(f"Server started on dynamically assigned port: {current_port}")
            break
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(
                    f"Port {current_port} already in use. Attempting to find a new port (attempt {attempt + 1}/{max_retries})."
                )
                current_port = 0  # Let the OS choose a random port
            else:
                raise
    else:
        raise RuntimeError(f"Failed to start Flask server after {max_retries} attempts.")

    # Update the redirect URL to reflect the actual port
    parsed_url = urlparse(broker_config.redirect_url)
    new_redirect_url = parsed_url._replace(netloc=f"{host}:{current_port}").geturl()
    logger.info(f"Using redirect URL: {new_redirect_url}")

    # Update the authenticator's redirect URL
    if global_authenticator_instance is None:
        raise RuntimeError("Authenticator instance not initialized")
    global_authenticator_instance.set_redirect_url(new_redirect_url)

    # Open the web UI instead of directly opening the OAuth URL
    # Use localhost for external access (outside container) but server binds to 0.0.0.0
    web_ui_url = f"http://localhost:{current_port}/"
    logger.info(f"🚀 Authentication UI is ready at: {web_ui_url}")
    logger.info(f"📱 Please open your browser and navigate to: {web_ui_url}")
    logger.info(f"🔧 Server binding: http://{server_host}:{current_port}/")

    # Try to open browser, but don't fail if it's not available (like in Docker)
    if auth_server_config.open_browser:
        try:
            webbrowser.open(web_ui_url)
            logger.info("✅ Browser opened successfully")
        except Exception as e:
            logger.warning(f"⚠️  Could not open browser automatically: {e}")
            logger.info(f"🔗 Please manually open: {web_ui_url}")

    try:
        # Wait for either success or error event
        auth_success_event.wait(timeout=auth_server_config.timeout_seconds)  # Wait for 5 minutes
        if not auth_success_event.is_set():
            logger.error("Authentication timed out or failed to complete.")
            auth_result["success"] = False
            auth_result["message"] = "Authentication timed out or failed."
    finally:
        logger.debug("Attempting to shut down Flask server.")
        server.shutdown()
        server.join()  # Ensure the server thread has completely shut down
        server.stopped.wait(timeout=5)  # Wait for the server to confirm it's stopped

    return bool(auth_result["success"])
