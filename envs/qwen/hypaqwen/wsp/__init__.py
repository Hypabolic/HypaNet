"""v0.4 weight-space programming (WSP) kernel (docs/V04_WSP_KERNEL.md).

* ``server``: the harness-owned model server (§3). It holds the weights, the edit state, the token
  meter, the edit journal and the ledger, and never runs author code.
* ``protocol``: the wire format and request schema shared by the server and the clients.
* ``author/``: the author-side kernel client (§4) and the ``./fwd`` tool. The harness copies this
  directory into the author dir as ``tools/hn_kernel/``. It imports only the standard library,
  numpy and torch, never ``transformers`` or ``hypaqwen``.

Importing this package is cheap; the server (and ``transformers``) load only from ``.server``.
"""
