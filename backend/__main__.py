"""Start the local backend with python -m backend."""

import uvicorn

from backend.api.app import create_app
from backend.config import load_config


def main():
    config = load_config()
    uvicorn.run(create_app(config), host=config["server"]["host"], port=config["server"]["port"],
                ws_max_size=config["server"].get("max_websocket_message_bytes", 9437184))


if __name__ == "__main__":
    main()
