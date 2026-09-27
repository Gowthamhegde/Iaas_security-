"""
Live flow feature extractor for IaaS servers.

Aggregates packets into bidirectional flows (5-tuple) and maps them to
NSL-KDD-compatible features so the trained Random Forest can score live traffic.

Requires: Scapy + (Windows: Npcap + admin) or (Linux: root / CAP_NET_RAW).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Destination port -> NSL-KDD-ish service label
_PORT_SERVICE = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 67: "dhcp", 68: "dhcp", 80: "http", 110: "pop_3",
    111: "sunrpc", 119: "nntp", 123: "ntp", 143: "imap4", 161: "snmp",
    443: "http_443", 445: "microsoft_ds", 993: "imap4", 995: "pop_3",
    1433: "sql_net", 3306: "sql_net", 3389: "remote_job", 5432: "sql_net",
    8080: "http_8001", 8443: "http_443",
}

FlowKey = Tuple[str, str, int, int, str]  # src_ip, dst_ip, sport, dport, proto


def _service_from_port(port: int) -> str:
    if port in _PORT_SERVICE:
        return _PORT_SERVICE[port]
    if port < 1024:
        return "private"
    return "other"


def _normalize_key(src_ip, dst_ip, sport, dport, proto) -> Tuple[FlowKey, bool]:
    """
    Canonicalize bidirectional flow key.
    Returns (key, swapped) where swapped=True if packet is reverse direction.
    """
    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a <= b:
        return (src_ip, dst_ip, sport, dport, proto), False
    return (dst_ip, src_ip, dport, sport, proto), True


@dataclass
class _FlowState:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_ts: float
    last_ts: float
    src_bytes: int = 0
    dst_bytes: int = 0
    src_pkts: int = 0
    dst_pkts: int = 0
    syn: int = 0
    ack: int = 0
    rst: int = 0
    fin: int = 0
    psh: int = 0
    urg: int = 0
    land: int = 0
    wrong_fragment: int = 0
    closed: bool = False

    def update(self, pkt_len: int, swapped: bool, tcp_flags=None, fragmented=False):
        now = time.time()
        self.last_ts = now
        if swapped:
            self.dst_bytes += pkt_len
            self.dst_pkts += 1
        else:
            self.src_bytes += pkt_len
            self.src_pkts += 1
        if fragmented:
            self.wrong_fragment += 1
        if tcp_flags is not None:
            if tcp_flags & 0x02:
                self.syn += 1
            if tcp_flags & 0x10:
                self.ack += 1
            if tcp_flags & 0x04:
                self.rst += 1
                self.closed = True
            if tcp_flags & 0x01:
                self.fin += 1
                if self.fin >= 2:
                    self.closed = True
            if tcp_flags & 0x08:
                self.psh += 1
            if tcp_flags & 0x20:
                self.urg += 1

    def tcp_flag_label(self) -> str:
        """Map observed TCP handshake/teardown pattern to NSL-KDD flag."""
        if self.protocol != "tcp":
            return "SF"
        if self.syn and not self.ack and self.rst:
            return "REJ"
        if self.syn and not self.ack and not self.rst and not self.fin:
            return "S0"
        if self.syn and self.ack and self.rst:
            return "RSTO"
        if self.syn and not self.ack and self.fin:
            return "SH"
        if self.syn == 1 and self.ack >= 1 and not self.fin and not self.rst:
            return "S1"
        if self.syn >= 2 and self.ack >= 1:
            return "S2"
        if self.fin and self.ack:
            return "SF"
        if self.rst:
            return "RSTR"
        return "OTH"


@dataclass
class _WindowStats:
    """Sliding connection window for count / host-based NSL-KDD features."""
    recent: Deque[dict] = field(default_factory=lambda: deque(maxlen=500))
    host_history: Dict[str, Deque[dict]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=100))
    )

    def add(self, rec: dict):
        self.recent.append(rec)
        self.host_history[rec["dst_ip"]].append(rec)

    def connection_features(self, dst_ip: str, service: str, now: float, window: float = 2.0):
        recent = [r for r in self.recent if now - r["ts"] <= window]
        same_host = [r for r in recent if r["dst_ip"] == dst_ip]
        same_srv = [r for r in recent if r["service"] == service]
        count = max(len(same_host), 1)
        srv_count = max(len(same_srv), 1)

        def _rate(rows, pred):
            if not rows:
                return 0.0
            return sum(1 for r in rows if pred(r)) / len(rows)

        serror = _rate(same_host, lambda r: r["flag"] in ("S0", "S1", "S2", "S3"))
        srv_serror = _rate(same_srv, lambda r: r["flag"] in ("S0", "S1", "S2", "S3"))
        rerror = _rate(same_host, lambda r: r["flag"] in ("REJ", "RSTO", "RSTR"))
        srv_rerror = _rate(same_srv, lambda r: r["flag"] in ("REJ", "RSTO", "RSTR"))
        same_srv_rate = len([r for r in same_host if r["service"] == service]) / count
        diff_srv_rate = 1.0 - same_srv_rate
        srv_diff_host = len({r["dst_ip"] for r in same_srv}) / srv_count

        host_rows = list(self.host_history[dst_ip])
        dst_host_count = max(len(host_rows), 1)
        dst_host_srv = [r for r in host_rows if r["service"] == service]
        dst_host_srv_count = max(len(dst_host_srv), 1)
        dst_host_same_srv = len(dst_host_srv) / dst_host_count
        dst_host_diff_srv = 1.0 - dst_host_same_srv
        same_src_port = len({r["src_port"] for r in host_rows})
        dst_host_same_src_port = (
            sum(1 for r in host_rows if r["src_port"] == host_rows[-1]["src_port"]) / dst_host_count
            if host_rows else 0.0
        )
        dst_host_srv_diff_host = (
            len({r["src_ip"] for r in dst_host_srv}) / dst_host_srv_count
        )
        dst_host_serror = _rate(host_rows, lambda r: r["flag"] in ("S0", "S1", "S2", "S3"))
        dst_host_srv_serror = _rate(dst_host_srv, lambda r: r["flag"] in ("S0", "S1", "S2", "S3"))
        dst_host_rerror = _rate(host_rows, lambda r: r["flag"] in ("REJ", "RSTO", "RSTR"))
        dst_host_srv_rerror = _rate(dst_host_srv, lambda r: r["flag"] in ("REJ", "RSTO", "RSTR"))

        return {
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror,
            "srv_serror_rate": srv_serror,
            "rerror_rate": rerror,
            "srv_rerror_rate": srv_rerror,
            "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate,
            "srv_diff_host_rate": min(srv_diff_host, 1.0),
            "dst_host_count": min(dst_host_count, 255),
            "dst_host_srv_count": min(dst_host_srv_count, 255),
            "dst_host_same_srv_rate": dst_host_same_srv,
            "dst_host_diff_srv_rate": dst_host_diff_srv,
            "dst_host_same_src_port_rate": dst_host_same_src_port,
            "dst_host_srv_diff_host_rate": min(dst_host_srv_diff_host, 1.0),
            "dst_host_serror_rate": dst_host_serror,
            "dst_host_srv_serror_rate": dst_host_srv_serror,
            "dst_host_rerror_rate": dst_host_rerror,
            "dst_host_srv_rerror_rate": dst_host_srv_rerror,
            "_same_src_port_uniq": same_src_port,
        }


class LiveFlowExtractor:
    """
    Capture packets on an interface, aggregate into flows, emit NSL-KDD-like dicts.

    Args:
        interface: NIC name (e.g. eth0, Ethernet). None = Scapy default.
        flow_timeout: idle seconds before a flow is exported
        bpf_filter: optional BPF filter string
        on_flow: callback(flow_dict) for each completed flow
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        flow_timeout: float = 30.0,
        bpf_filter: str = "ip",
        on_flow: Optional[Callable[[dict], None]] = None,
    ):
        self.interface = interface
        self.flow_timeout = flow_timeout
        self.bpf_filter = bpf_filter
        self.on_flow = on_flow
        self._flows: Dict[FlowKey, _FlowState] = {}
        self._lock = threading.Lock()
        self._window = _WindowStats()
        self._running = False
        self._sniff_thread: Optional[threading.Thread] = None
        self._reaper_thread: Optional[threading.Thread] = None

    def _emit(self, state: _FlowState):
        now = time.time()
        duration = max(int(state.last_ts - state.start_ts), 0)
        service = _service_from_port(state.dst_port)
        flag = state.tcp_flag_label()
        logged_in = 1 if (
            state.protocol == "tcp" and state.syn and state.ack and state.src_bytes > 0
        ) else 0

        win = self._window.connection_features(state.dst_ip, service, now)

        flow = {
            "duration": duration,
            "protocol_type": state.protocol,
            "service": service,
            "flag": flag,
            "src_bytes": state.src_bytes,
            "dst_bytes": state.dst_bytes,
            "land": state.land,
            "wrong_fragment": min(state.wrong_fragment, 3),
            "urgent": min(state.urg, 3),
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": logged_in,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": win["count"],
            "srv_count": win["srv_count"],
            "serror_rate": win["serror_rate"],
            "srv_serror_rate": win["srv_serror_rate"],
            "rerror_rate": win["rerror_rate"],
            "srv_rerror_rate": win["srv_rerror_rate"],
            "same_srv_rate": win["same_srv_rate"],
            "diff_srv_rate": win["diff_srv_rate"],
            "srv_diff_host_rate": win["srv_diff_host_rate"],
            "dst_host_count": win["dst_host_count"],
            "dst_host_srv_count": win["dst_host_srv_count"],
            "dst_host_same_srv_rate": win["dst_host_same_srv_rate"],
            "dst_host_diff_srv_rate": win["dst_host_diff_srv_rate"],
            "dst_host_same_src_port_rate": win["dst_host_same_src_port_rate"],
            "dst_host_srv_diff_host_rate": win["dst_host_srv_diff_host_rate"],
            "dst_host_serror_rate": win["dst_host_serror_rate"],
            "dst_host_srv_serror_rate": win["dst_host_srv_serror_rate"],
            "dst_host_rerror_rate": win["dst_host_rerror_rate"],
            "dst_host_srv_rerror_rate": win["dst_host_srv_rerror_rate"],
            "_true_label": "Unknown",
            "_timestamp": datetime.now().isoformat(),
            "_src_ip": state.src_ip,
            "_dst_ip": state.dst_ip,
            "_src_port": state.src_port,
            "_dst_port": state.dst_port,
            "_capture_mode": "live",
        }

        self._window.add({
            "ts": now,
            "dst_ip": state.dst_ip,
            "src_ip": state.src_ip,
            "src_port": state.src_port,
            "service": service,
            "flag": flag,
        })

        if self.on_flow:
            try:
                self.on_flow(flow)
            except Exception as e:
                logger.error(f"[FlowExtractor] on_flow callback failed: {e}")

    def _handle_packet(self, pkt):
        try:
            from scapy.all import IP, TCP, UDP, ICMP, Raw  # noqa: F401
        except ImportError:
            return

        if IP not in pkt:
            return
        ip = pkt[IP]
        proto = "tcp" if TCP in pkt else "udp" if UDP in pkt else "icmp" if ICMP in pkt else None
        if proto is None:
            return

        sport = int(pkt[TCP].sport) if TCP in pkt else (int(pkt[UDP].sport) if UDP in pkt else 0)
        dport = int(pkt[TCP].dport) if TCP in pkt else (int(pkt[UDP].dport) if UDP in pkt else 0)
        key, _ = _normalize_key(ip.src, ip.dst, sport, dport, proto)
        pkt_len = int(ip.len) if hasattr(ip, "len") and ip.len else len(bytes(pkt))
        fragmented = bool(getattr(ip, "frag", 0) or (getattr(ip, "flags", 0) & 0x1))
        tcp_flags = int(pkt[TCP].flags) if TCP in pkt else None

        with self._lock:
            state = self._flows.get(key)
            if state is None:
                # Orient as client -> server (prefer well-known destination port)
                if sport and sport < 1024 and dport >= 1024:
                    src_ip, dst_ip, sport_m, dport_m = ip.dst, ip.src, dport, sport
                else:
                    src_ip, dst_ip, sport_m, dport_m = ip.src, ip.dst, sport, dport
                now = time.time()
                state = _FlowState(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=sport_m,
                    dst_port=dport_m,
                    protocol=proto,
                    start_ts=now,
                    last_ts=now,
                    land=1 if (ip.src == ip.dst and sport == dport) else 0,
                )
                self._flows[key] = state

            from_src = (ip.src == state.src_ip)
            state.update(
                pkt_len,
                swapped=not from_src,
                tcp_flags=tcp_flags,
                fragmented=fragmented,
            )
            if state.closed:
                self._emit(state)
                del self._flows[key]

    def _reap_idle(self):
        while self._running:
            time.sleep(1.0)
            now = time.time()
            with self._lock:
                expired = [
                    k for k, s in self._flows.items()
                    if (now - s.last_ts) >= self.flow_timeout or s.closed
                ]
                for k in expired:
                    self._emit(self._flows[k])
                    del self._flows[k]

    def _sniff_loop(self):
        try:
            from scapy.all import sniff
        except ImportError as e:
            logger.error(f"[FlowExtractor] Scapy not installed: {e}")
            self._running = False
            return

        kwargs = {
            "prn": self._handle_packet,
            "store": False,
            "filter": self.bpf_filter,
            "stop_filter": lambda _: not self._running,
        }
        if self.interface:
            kwargs["iface"] = self.interface

        logger.info(
            f"[FlowExtractor] Live capture started "
            f"(iface={self.interface or 'default'}, timeout={self.flow_timeout}s)"
        )
        try:
            sniff(**kwargs)
        except Exception as e:
            logger.error(
                f"[FlowExtractor] Capture failed: {e}. "
                "On Windows install Npcap and run as Administrator; "
                "on Linux run with root or CAP_NET_RAW."
            )
            self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._sniff_thread = threading.Thread(
            target=self._sniff_loop, daemon=True, name="flow-sniff"
        )
        self._reaper_thread = threading.Thread(
            target=self._reap_idle, daemon=True, name="flow-reaper"
        )
        self._sniff_thread.start()
        self._reaper_thread.start()

    def stop(self):
        self._running = False
        # Flush remaining flows
        with self._lock:
            for k in list(self._flows.keys()):
                self._emit(self._flows[k])
                del self._flows[k]
        if self._sniff_thread:
            self._sniff_thread.join(timeout=3)
        if self._reaper_thread:
            self._reaper_thread.join(timeout=3)
        logger.info("[FlowExtractor] Stopped.")

    @staticmethod
    def list_interfaces() -> List[str]:
        try:
            from scapy.all import get_if_list
            return list(get_if_list())
        except Exception:
            return []
