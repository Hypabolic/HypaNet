"""Server-side view of the WSP wire format (docs/V04_WSP_KERNEL.md §3).

The implementation lives in ``author/protocol.py``, which is self-contained (stdlib + numpy) because
it is copied into the author directory. Re-exporting it here keeps one source for both ends.
"""

from .author.protocol import (  # noqa: F401
    MAGIC,
    MAX_HEADER_BYTES,
    MAX_TENSOR_BYTES,
    MAX_TOPK,
    MODELS,
    N_LAYERS,
    WIRE_DTYPE,
    ProtocolError,
    RequestError,
    connect,
    error_header,
    listen,
    recv_msg,
    send_msg,
    validate_disclosed_request,
    validate_forward_request,
)
