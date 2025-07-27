import os
import threading
import time
import webbrowser
from datetime import datetime
from typing import Optional, Union
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
def auth_status() -> Union[Response, tuple[Response, int]]:
    """API endpoint to check current token status."""
    logger.debug("Received request for /auth/status")
    global global_token_manager_instance

    if global_token_manager_instance is None:
        logger.error("Auth status check failed: TokenManager has not been initialized.")
        return jsonify({"valid": False, "error": "CRITICAL: Token manager not initialized"}), 500

    try:
        token = global_token_manager_instance.get_access_token()
        if global_token_manager_instance.is_token_available():
            token_preview = f"{token[:10]}..." if token and len(token) > 10 else "Token available"
            logger.info(f"Auth status check: Token is available. Preview: {token_preview}")
            return jsonify(
                {
                    "valid": True,
                    "token_preview": token_preview,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        logger.warning("Auth status check: No valid access token found.")
        return jsonify({"valid": False, "error": "No access token found"})
    except Exception as e:
        logger.error(f"An unexpected error occurred in auth_status: {e}", exc_info=True)
        return jsonify({"valid": False, "error": "An internal server error occurred."}), 500


@app.route("/auth/login")
def auth_login() -> str:
    """Initiate the Zerodha OAuth flow."""
    logger.info("Received request for /auth/login, initiating OAuth flow.")
    global global_authenticator_instance

    if global_authenticator_instance is None:
        logger.critical("Cannot initiate login: Authenticator has not been initialized.")
        return render_template(
            "auth_error.html",
            error_type="Configuration Error",
            error_message="CRITICAL: Authenticator not initialized. System cannot proceed.",
        )

    try:
        auth_url = global_authenticator_instance.generate_login_url()
        logger.info(f"Redirecting user to Zerodha OAuth URL: {auth_url}")
        # This is a simple redirect page, no major changes needed here.
        # The core logic is in the URL generation and the callback.
        return f'''
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
        '''
    except (RuntimeError, ValueError) as e:
        logger.critical(f"Failed to generate login URL due to a critical error: {e}", exc_info=True)
        return render_template("auth_error.html", error_type="Login Generation Error", error_message=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during login URL generation: {e}", exc_info=True)
        return render_template("auth_error.html", error_type="Unexpected Login Error", error_message=str(e))


@app.route("/auth/refresh")
def auth_refresh() -> str:
    """Refresh token by reinitializing the OAuth flow."""
    return auth_login()


@app.route("/callback")
def auth_callback() -> str:
    """Handles the OAuth callback from Zerodha after user authentication."""
    global auth_result, global_authenticator_instance, global_token_manager_instance

    logger.info(f"Received callback from broker with args: {request.args}")
    request_token = request.args.get("request_token")
    status = request.args.get("status")

    if status != "success" or not request_token:
        error = request.args.get("error", "Unknown Error")
        error_message = request.args.get("message", "No error message provided.")
        logger.critical(f"OAuth failed from broker. Status: {status}, Error: {error}, Message: {error_message}")
        auth_result = {"success": False, "message": f"OAuth failed: {error} - {error_message}"}
        auth_error_event.set()
        return render_template(
            "auth_error.html",
            error_type=error,
            error_message=error_message,
            request_token=request_token,
        )

    logger.info("OAuth login successful, proceeding to token exchange.")
    try:
        if global_authenticator_instance is None or global_token_manager_instance is None:
            logger.critical("Callback cannot proceed: Authenticator or TokenManager not initialized.")
            raise RuntimeError("CRITICAL: System components not initialized for callback.")

        logger.info("About to call authenticate() with request token")
        access_token = global_authenticator_instance.authenticate(request_token)
        logger.info(f"Authentication successful. Got access_token: {'Yes' if access_token else 'No'}")

        logger.info("About to call set_token()")
        try:
            global_token_manager_instance.set_token(access_token)
            logger.info("set_token() completed successfully")

        except Exception as token_error:
            logger.critical(f"CRITICAL ERROR in set_token(): {token_error}", exc_info=True)
            auth_error_event.set()
            return render_template(
                "auth_error.html",
                error_type="TOKEN_SAVE_ERROR",
                error_message=str(token_error),
                request_token=request_token,
            )

        # CRITICAL FIX: Set the success event immediately after token is saved
        # Use try-catch to ensure event setting doesn't fail silently
        try:
            logger.info("Setting auth_success_event...")
            auth_success_event.set()
            logger.info(
                f"Authentication flow completed successfully. Success event has been set! Event state: {auth_success_event.is_set()}"
            )
        except Exception as event_error:
            logger.critical(f"CRITICAL ERROR setting success event: {event_error}", exc_info=True)
            # Continue anyway since token is already saved

        # Also update the global auth_result for debugging
        auth_result = {"success": True, "message": "Authentication completed successfully"}

        token_preview = f"{access_token[:10]}..." if len(access_token) > 10 else access_token
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template(
            "auth_success.html", token_preview=token_preview, timestamp=timestamp, auto_close_seconds=5
        )

    except (RuntimeError, ValueError) as e:
        logger.critical(f"A critical error occurred during token exchange: {e}", exc_info=True)
        auth_result = {"success": False, "message": str(e)}
        auth_error_event.set()
        return render_template("auth_error.html", error_type="Token Exchange Error", error_message=str(e))
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during callback processing: {e}", exc_info=True)
        auth_result = {"success": False, "message": f"An unexpected error occurred: {e}"}
        auth_error_event.set()
        return render_template("auth_error.html", error_type="Unexpected Server Error", error_message=str(e))


def start_auth_server_and_wait(broker_config: BrokerConfig, auth_server_config: AuthServerConfig) -> bool:
    global global_token_manager_instance, global_authenticator_instance
    global auth_success_event, auth_error_event, auth_result

    logger.info("Initializing authentication process...")

    # CRITICAL FIX: Reset global authentication events and result
    logger.info("Resetting authentication events from previous attempts...")
    auth_success_event.clear()
    auth_error_event.clear()
    auth_result = {"success": False, "message": ""}
    logger.info("Authentication events reset complete")

    try:
        global_token_manager_instance = TokenManager()
        global_authenticator_instance = KiteAuthenticator(broker_config)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Failed to initialize core authentication components: {e}", exc_info=True)
        return False

    if not broker_config.redirect_url:
        logger.critical("CRITICAL: Broker redirect URL is not configured in config.yaml.")
        raise ValueError("Broker redirect URL is mandatory for the authentication server.")

    try:
        parsed_url = urlparse(broker_config.redirect_url)
        host: Optional[str] = parsed_url.hostname
        initial_port: Optional[int] = parsed_url.port
        if not host or not initial_port:
            raise ValueError("Hostname or port is missing in the redirect URL.")
    except (ValueError, AttributeError) as e:
        logger.critical(f"Invalid redirect URL configured: '{broker_config.redirect_url}'. {e}")
        raise ValueError(f"Invalid redirect URL: {broker_config.redirect_url}") from e

    server: Optional[FlaskServer] = None
    current_port = initial_port
    server_host = auth_server_config.server_host
    if not server_host:
        logger.critical("CRITICAL: Auth server host is not configured in config.yaml.")
        raise ValueError("auth_server.server_host is a mandatory configuration.")

    for attempt in range(auth_server_config.max_retries):
        try:
            logger.info(f"Attempting to start auth server on {server_host}:{current_port} (Attempt {attempt + 1})")
            server = FlaskServer(server_host, current_port)
            server.daemon = True
            server.start()
            if current_port == 0:
                # Get the actual port if 0 was used for dynamic allocation
                current_port = server.srv.socket.getsockname()[1]
                logger.info(f"Server started on dynamically assigned port: {current_port}")
            break  # Exit loop on success
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {current_port} is in use. Retrying on a different port.")
                current_port = 0  # Let the OS choose a random available port
            else:
                logger.critical(f"An unexpected OS error occurred while starting the server: {e}", exc_info=True)
                raise RuntimeError("Failed to start auth server due to an OS error.") from e
        except Exception as e:
            logger.critical(f"An unexpected error occurred while starting the server: {e}", exc_info=True)
            raise RuntimeError("An unexpected error prevented the auth server from starting.") from e
    else:
        logger.critical(f"Failed to start Flask server after {auth_server_config.max_retries} attempts.")
        raise RuntimeError("Unable to secure a port for the authentication server.")

    # Update redirect URL and authenticator with the actual port used
    new_redirect_url = urlparse(broker_config.redirect_url)._replace(netloc=f"{host}:{current_port}").geturl()
    logger.info(f"Final redirect URL for OAuth flow: {new_redirect_url}")
    global_authenticator_instance.set_redirect_url(new_redirect_url)

    web_ui_url = f"http://localhost:{current_port}/"
    logger.info(f"🚀 Authentication UI is ready at: {web_ui_url}")
    logger.info(f"🔧 Server is bound to: http://{server_host}:{current_port}/")

    if auth_server_config.open_browser:
        logger.info("Attempting to open web browser automatically.")
        try:
            webbrowser.open(web_ui_url)
            logger.info("✅ Successfully opened web browser.")
        except Exception as e:
            logger.error(f"⚠️ Could not open browser automatically: {e}. Please open the URL manually.", exc_info=True)
            logger.info(f"🔗 Please manually navigate to: {web_ui_url}")

    try:
        timeout = auth_server_config.timeout_seconds
        logger.info(f"Waiting for authentication to complete. Timeout is set to {timeout} seconds.")

        # Wait for either success or error event
        check_count = 0
        while timeout > 0:
            check_count += 1

            # Log periodic status every 10 seconds
            if check_count % 10 == 0:
                logger.info(
                    f"⏰ Authentication status check #{check_count}: {timeout}s remaining, success={auth_success_event.is_set()}, error={auth_error_event.is_set()}"
                )

            if auth_success_event.is_set():
                logger.info("🎉 Authentication completed successfully!")
                # Ensure server is shut down before returning
                if server:
                    server.shutdown()
                    server.join(timeout=5)
                    if server.is_alive():
                        logger.warning("Server thread did not shut down gracefully.")
                return True
            if auth_error_event.is_set():
                logger.error("❌ Authentication failed with error.")
                return False

            # FALLBACK: Check if token was actually saved (in case event mechanism fails)
            if global_token_manager_instance and global_token_manager_instance.is_token_available():
                logger.warning("🔍 Token detected via fallback check - event mechanism may have failed")
                logger.info("🎉 Authentication completed successfully (via fallback detection)!")
                # Ensure server is shut down before returning
                if server:
                    server.shutdown()
                    server.join(timeout=5)
                    if server.is_alive():
                        logger.warning("Server thread did not shut down gracefully.")
                return True
            time.sleep(1)  # Using time.sleep instead of await asyncio.sleep
            timeout -= 1

        logger.error(f"Authentication timed out after {auth_server_config.timeout_seconds} seconds.")
        return False

    finally:
        logger.info("Authentication process finished. Shutting down the web server.")
        if server and server.is_alive():
            server.shutdown()
            server.join(timeout=5)
            if server.is_alive():
                logger.warning("Server thread did not shut down gracefully.")
