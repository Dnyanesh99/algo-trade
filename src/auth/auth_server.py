import threading
import webbrowser
from typing import Optional
from urllib.parse import urlparse

from flask import Flask, request
from werkzeug.serving import make_server

from src.auth.authenticator import KiteAuthenticator
from src.auth.token_manager import TokenManager
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()

app = Flask(__name__)
auth_success_event = threading.Event()
auth_error_event = threading.Event()
auth_result = {"success": False, "message": ""}

# Global instances for authenticator and token manager
# These must be defined at the module level to be accessible globally
global_token_manager_instance = TokenManager()
global_authenticator_instance = KiteAuthenticator(global_token_manager_instance)


class FlaskServer(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self) -> None:
        logger.info(f"Starting Flask server on http://{self.host}:{self.port}")
        self.srv.serve_forever()

    def shutdown(self) -> None:
        logger.info("Shutting down Flask server.")
        self.srv.shutdown()


@app.route("/auth")  # type: ignore[misc]
def auth_callback() -> str:
    global auth_result
    global global_authenticator_instance
    global global_token_manager_instance
    request_token = request.args.get("request_token")
    status = request.args.get("status")
    error = request.args.get("error")
    error_message = request.args.get("message")

    if status == "success" and request_token:
        logger.info(f"Received request token: {request_token}")
        try:
            # Use the global authenticator instance
            global_authenticator_instance.authenticate(request_token)
            auth_result["success"] = True
            auth_result["message"] = "Authentication successful! You can close this window."
            logger.info("Authentication flow completed successfully.")
            auth_success_event.set()
            return "<h1>Authentication Successful!</h1><p>You can close this window.</p>"
        except Exception as e:
            auth_result["success"] = False
            auth_result["message"] = f"Authentication failed: {e}"
            logger.error(f"Authentication failed during token exchange: {e}")
            auth_error_event.set()
            return f"<h1>Authentication Failed!</h1><p>Error: {e}</p>"
    else:
        auth_result["success"] = False
        auth_result["message"] = f"Authentication failed: {error} - {error_message}"
        logger.error(f"Authentication failed from Kite: {error} - {error_message}")
        auth_error_event.set()
        return f"<h1>Authentication Failed!</h1><p>Error: {error} - {error_message}</p>"


def start_auth_server_and_wait() -> bool:
    host: Optional[str] = urlparse(config.broker.redirect_url).hostname
    port: Optional[int] = urlparse(config.broker.redirect_url).port

    if host is None or port is None:
        raise ValueError(f"Invalid redirect URL configured: {config.broker.redirect_url}. Host or port is missing.")

    server = FlaskServer(host, port)
    server.daemon = True  # Allow the main program to exit even if the server is running
    server.start()

    # Use the global authenticator instance
    auth_url = global_authenticator_instance.generate_login_url()
    logger.info(f"Opening browser for authentication: {auth_url}")
    webbrowser.open(auth_url)

    # Wait for either success or error event
    auth_success_event.wait(timeout=300)  # Wait for 5 minutes
    if not auth_success_event.is_set():
        logger.error("Authentication timed out or failed to complete.")
        auth_result["success"] = False
        auth_result["message"] = "Authentication timed out or failed."

    server.shutdown()
    server.join()  # Ensure the server thread has completely shut down

    return bool(auth_result["success"])
