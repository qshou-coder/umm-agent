#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time


def _now() -> str:
    return time.strftime("%F %T")


def _log(msg: str) -> None:
    print(f"[preflight] {_now()} {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed launch preflight check.")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", type=str, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--check-port-offset", type=int, default=17)
    return parser.parse_args()


def _validate_args(args):
    if args.nnodes <= 0:
        raise ValueError(f"nnodes must be > 0, got {args.nnodes}")
    if not (0 <= args.node_rank < args.nnodes):
        raise ValueError(
            f"node_rank must be in [0, {args.nnodes - 1}], got {args.node_rank}"
        )
    if not (1 <= args.master_port <= 65535):
        raise ValueError(f"master_port must be in [1, 65535], got {args.master_port}")


def _recv_line(conn: socket.socket, timeout_sec: float) -> str:
    conn.settimeout(timeout_sec)
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode("utf-8", errors="replace").strip()


def _send_line(conn: socket.socket, payload: dict):
    msg = json.dumps(payload, ensure_ascii=False) + "\n"
    conn.sendall(msg.encode("utf-8"))


def run_server(args, check_port: int) -> int:
    deadline = time.time() + args.timeout_sec
    hostname = socket.gethostname()
    peers = {args.node_rank: hostname}
    conns = []

    _log(
        f"rank0 listening on {args.master_addr}:{check_port} "
        f"(expect nnodes={args.nnodes})"
    )

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("0.0.0.0", check_port))
    listen_sock.listen(max(8, args.nnodes))
    listen_sock.settimeout(1.0)

    duplicate_rank = None
    errors = []
    try:
        while time.time() < deadline and len(peers) < args.nnodes:
            try:
                conn, addr = listen_sock.accept()
            except socket.timeout:
                continue
            try:
                line = _recv_line(conn, timeout_sec=3.0)
                payload = json.loads(line)
                remote_rank = int(payload["node_rank"])
                remote_host = payload.get("hostname", str(addr[0]))
                if remote_rank in peers:
                    duplicate_rank = remote_rank
                    errors.append(
                        f"duplicate node_rank={remote_rank}: "
                        f"{peers[remote_rank]} and {remote_host}"
                    )
                else:
                    peers[remote_rank] = remote_host
                conns.append(conn)
            except Exception as e:
                errors.append(f"bad handshake from {addr}: {e}")
                try:
                    conn.close()
                except Exception:
                    pass

        ok = True
        if duplicate_rank is not None:
            ok = False
        if len(peers) != args.nnodes:
            ok = False
            missing = sorted(set(range(args.nnodes)) - set(peers.keys()))
            errors.append(f"missing node_ranks={missing}")

        result = {"ok": ok, "peers": peers, "errors": errors}
        for conn in conns:
            try:
                _send_line(conn, result)
                conn.close()
            except Exception:
                pass

        if ok:
            _log(f"PASS peers={peers}")
            return 0
        _log(f"FAIL peers={peers} errors={errors}")
        return 2
    finally:
        listen_sock.close()


def run_client(args, check_port: int) -> int:
    hostname = socket.gethostname()
    deadline = time.time() + args.timeout_sec
    payload = {"node_rank": args.node_rank, "hostname": hostname}
    _log(
        f"rank{args.node_rank} connecting to {args.master_addr}:{check_port} "
        f"(timeout={args.timeout_sec}s)"
    )

    while time.time() < deadline:
        try:
            with socket.create_connection(
                (args.master_addr, check_port), timeout=3.0
            ) as conn:
                _send_line(conn, payload)
                line = _recv_line(conn, timeout_sec=max(3.0, args.timeout_sec))
                result = json.loads(line)
                if result.get("ok", False):
                    _log(f"PASS ack from rank0 peers={result.get('peers', {})}")
                    return 0
                _log(f"FAIL ack from rank0: {result}")
                return 2
        except OSError:
            time.sleep(0.5)
        except Exception as e:
            _log(f"FAIL bad response from rank0: {e}")
            return 2

    _log("FAIL timeout waiting for rank0 preflight server")
    return 2


def main():
    args = parse_args()
    try:
        _validate_args(args)
    except Exception as e:
        _log(f"FAIL invalid args: {e}")
        return 2

    check_port = args.master_port + args.check_port_offset
    if check_port > 65535:
        _log(
            f"FAIL check_port out of range: master_port({args.master_port}) + "
            f"offset({args.check_port_offset})"
        )
        return 2

    try:
        socket.gethostbyname(args.master_addr)
    except Exception as e:
        _log(f"FAIL cannot resolve master_addr={args.master_addr}: {e}")
        return 2

    if args.node_rank == 0:
        return run_server(args, check_port)
    return run_client(args, check_port)


if __name__ == "__main__":
    sys.exit(main())

