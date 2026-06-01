import logging
def preprocess(app):
    app.add_middleware(LoggingMiddleware)

class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)

logger = logging.getLogger(__name__)

def log_request(input_data):
    logger.info(f"Incoming request: {input_data}")


def register(server):
    server.add_middleware(LoggingMiddleware)
    server.register_hook("pre", log_request)

