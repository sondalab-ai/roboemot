"""Entry point used by the Reachy Mini daemon (``python -m roboemot_reachymini.main``)."""

from .app import RoboEmotApp

if __name__ == "__main__":
    app = RoboEmotApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
