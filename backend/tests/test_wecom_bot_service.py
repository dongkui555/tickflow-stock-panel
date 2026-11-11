import asyncio
import json

import pytest

from app.services.wecom_bot_service import WecomBotService


@pytest.mark.parametrize("code,status", [(0, "sent"), (40001, "failed"), (None, "unknown"), ("0", "unknown")])
def test_active_send_requires_matching_receipt(code, status):
    async def run():
        service = WecomBotService()
        service._connected = True
        frames = []
        class Socket:
            async def send(self, raw):
                frame = json.loads(raw)
                frames.append(frame)
                service._log_incoming(json.dumps({"headers": {"req_id": "unrelated"}, "errcode": 0}))
                assert not next(iter(service._pending.values())).done()
                service._log_incoming(json.dumps({"headers": frame["headers"], "errcode": code}))
        service._ws = Socket()
        result = await service._send_markdown("user1", "prediction", .05)
        assert result["status"] == status
        assert frames[0]["cmd"] == "aibot_send_msg"
        assert frames[0]["body"] == {"chatid": "user1", "msgtype": "markdown", "markdown": {"content": "prediction"}}
        assert not service._pending
    asyncio.run(run())


def test_timeout_is_unknown_not_success_or_retryable_failure():
    async def run():
        service = WecomBotService()
        service._connected = True
        class Socket:
            async def send(self, raw):
                pass
        service._ws = Socket()
        result = await service._send_markdown("user1", "prediction", .01)
        assert result["status"] == "unknown"
        assert not service._pending
    asyncio.run(run())


def test_disconnect_resolves_pending_send_as_unknown():
    async def run():
        service = WecomBotService()
        service._connected = True
        class Socket:
            async def send(self, raw):
                service._disconnect_pending()
        service._ws = Socket()
        assert (await service._send_markdown("user1", "prediction", .01))["status"] == "unknown"
    asyncio.run(run())


def test_known_chats_are_bot_bound_and_never_auto_subscribed(monkeypatch, tmp_path):
    service = WecomBotService()
    path = tmp_path / "chats.json"
    monkeypatch.setattr(service, "_chat_path", lambda: path)
    monkeypatch.setattr("app.services.preferences.get_wecom_bot_id", lambda: "bot1")
    saved = []
    monkeypatch.setattr("app.services.preferences.save", saved.append)
    for body in ({"chattype": "single", "from": {"userid": "user1"}},
                 {"chattype": "group", "chatid": "group1", "from": {"userid": "user1"}}):
        service._log_incoming(json.dumps({"cmd": "aibot_msg_callback", "body": body}))
    chats = service._known_chats()
    assert {c["chatid"] for c in chats} == {"user1", "group1"}
    assert not saved
    assert "content" not in path.read_text()
    monkeypatch.setattr("app.services.preferences.get_wecom_bot_id", lambda: "bot2")
    assert service._known_chats() == []


def test_heartbeats_are_not_starved_by_incoming_messages(monkeypatch):
    import app.services.wecom_bot_service as module
    monkeypatch.setattr(module, "_HEARTBEAT_INTERVAL", .01)
    async def run():
        service = WecomBotService()
        service._running = True
        sent = []
        class Socket:
            async def recv(self):
                await asyncio.sleep(.002)
                return '{}'
            async def send(self, raw):
                sent.append(json.loads(raw))
                service._running = False
        await asyncio.wait_for(service._maintain_connection(Socket()), .2)
        assert sent[0]["cmd"] == "ping"
        assert sent[0]["headers"]["req_id"]
    asyncio.run(run())
