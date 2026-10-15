"""TEST-ONLY network denial (spec 17.3). Loaded through PYTHONPATH by the black-box harness; no production module imports or disables it."""
import socket


class NetworkDenied(OSError):
    pass


def _deny(*_a, **_k):
    raise NetworkDenied("network access denied by the black-box harness")


socket.getaddrinfo = _deny
socket.gethostbyname = _deny
socket.gethostbyname_ex = _deny
socket.create_connection = _deny
socket.socket.connect = _deny
socket.socket.connect_ex = _deny
DENY_ACTIVE = True
