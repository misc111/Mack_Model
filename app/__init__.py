from typing import Any

from flask import Flask

from .routes import main_bp


def create_app(config_object: Any = "config.Config") -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(config_object)
    app.register_blueprint(main_bp)
    return app
