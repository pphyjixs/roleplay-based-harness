"""Bound inline media request bodies before JSON parsing or image decoding."""
import json


class MessageBodyLimit:
    def __init__(self, app, maximum):
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in {"/api/rp/messages", "/api/rp/message"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.maximum:
                output = json.dumps({"detail": "Message exceeds vision.max_request_bytes"}).encode()
                await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": output})
                return
            if not event.get("more_body", False):
                break
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)
