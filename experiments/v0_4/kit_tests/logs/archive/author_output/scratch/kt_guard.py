import os
def attempt(label, fn):
    try:
        fn()
    except BaseException as e:
        print('KT_GUARD', label, 'refused', type(e).__name__)
    else:
        print('KT_GUARD', label, 'ALLOWED')
attempt('read_outside', lambda: open('/etc/hosts').read())
attempt('read_parent', lambda: os.listdir('..'))
attempt('write_root', lambda: open('kt_root.txt', 'w'))
attempt('write_fwd', lambda: open('output/wsp-a/fwd/kt_forged.json', 'w'))
def fwd_socket():
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('.kernel/fwd.sock')
attempt('fwd_socket', fwd_socket)
def proc():
    import subprocess
    subprocess.run(['ls'])
attempt('subprocess', proc)
def optim():
    import torch.optim
attempt('torch_optim', optim)
def backward():
    import torch
    x = torch.ones(2, requires_grad=True)
    (x * x).sum().backward()
attempt('backward', backward)
def rng():
    import random
    random.random()
attempt('rng', rng)
print('KT_GUARD_DONE')
