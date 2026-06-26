bl_info = {
    "name": "Phone Camera Control",
    "author": "Custom",
    "version": (1, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Phone Cam",
    "description": "Control Blender camera with smartphone gyroscope via WebSocket",
    "category": "Camera",
}

import bpy
import asyncio
import threading
import json
import math
import socket
from mathutils import Quaternion, Euler, Matrix

# ──────────────────────────────────────────────
#  Simple WebSocket server (no external deps)
# ──────────────────────────────────────────────

def _ws_handshake(conn):
    """Perform WebSocket HTTP upgrade handshake."""
    import hashlib, base64
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            return False
        data += chunk

    key = None
    for line in data.decode("utf-8", errors="replace").split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
            break

    if not key:
        return False

    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(
        hashlib.sha1((key + magic).encode()).digest()
    ).decode()

    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    conn.sendall(response.encode())
    return True


def _ws_recv_frame(conn):
    """Read one WebSocket frame and return payload string (or None on error)."""
    try:
        header = b""
        while len(header) < 2:
            b = conn.recv(2 - len(header))
            if not b:
                return None
            header += b

        b1, b2 = header[0], header[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F

        if opcode == 8:   # close frame
            return None
        if opcode not in (1, 2):  # only text/binary
            return None

        if length == 126:
            raw = b""
            while len(raw) < 2:
                raw += conn.recv(2 - len(raw))
            length = int.from_bytes(raw, "big")
        elif length == 127:
            raw = b""
            while len(raw) < 8:
                raw += conn.recv(8 - len(raw))
            length = int.from_bytes(raw, "big")

        mask_key = b""
        if masked:
            while len(mask_key) < 4:
                mask_key += conn.recv(4 - len(mask_key))

        payload = b""
        while len(payload) < length:
            chunk = conn.recv(length - len(payload))
            if not chunk:
                return None
            payload += chunk

        if masked:
            payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(len(payload)))

        return payload.decode("utf-8", errors="replace")
    except Exception:
        return None


# ──────────────────────────────────────────────
#  Global state
# ──────────────────────────────────────────────

_server_thread = None
_server_socket = None
_running = False
_latest_data = {}          # shared between threads
_data_lock = threading.Lock()
_client_count = 0


def _client_handler(conn, addr):
    global _client_count, _latest_data
    if not _ws_handshake(conn):
        conn.close()
        return

    _client_count += 1
    print(f"[PhoneCam] Client connected: {addr}  (total: {_client_count})")

    try:
        while _running:
            msg = _ws_recv_frame(conn)
            if msg is None:
                break
            try:
                data = json.loads(msg)
                with _data_lock:
                    _latest_data.update(data)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"[PhoneCam] Client error: {e}")
    finally:
        _client_count -= 1
        print(f"[PhoneCam] Client disconnected: {addr}  (total: {_client_count})")
        conn.close()


def _server_loop(port):
    global _server_socket, _running
    _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        _server_socket.bind(("0.0.0.0", port))
        _server_socket.listen(5)
        _server_socket.settimeout(1.0)
        print(f"[PhoneCam] Server started on port {port}")

        while _running:
            try:
                conn, addr = _server_socket.accept()
                t = threading.Thread(target=_client_handler, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
    except Exception as e:
        print(f"[PhoneCam] Server error: {e}")
    finally:
        _server_socket.close()
        print("[PhoneCam] Server stopped")


def start_server(port=8765):
    global _server_thread, _running
    if _running:
        return
    _running = True
    _server_thread = threading.Thread(target=_server_loop, args=(port,), daemon=True)
    _server_thread.start()


def stop_server():
    global _running
    _running = False


# ──────────────────────────────────────────────
#  Blender timer – applies data every frame
# ──────────────────────────────────────────────

def _apply_rotation():
    props = bpy.context.scene.phone_cam_props

    with _data_lock:
        data = dict(_latest_data)

    if not data:
        return props.update_interval

    # ── choose target object ──
    cam_name = props.camera_name
    obj = bpy.data.objects.get(cam_name) if cam_name else None
    if obj is None:
        # fallback: active camera
        scene_cam = bpy.context.scene.camera
        if scene_cam:
            obj = scene_cam

    if obj is None:
        return props.update_interval

    alpha = props.smoothing  # 0 = instant, 0.99 = very smooth

    # ── Quaternion mode (preferred – best accuracy) ──
    if "qw" in data and "qx" in data:
        qw = data["qw"]
        qx = data["qx"]
        qy = data["qy"]
        qz = data["qz"]

        # Phone: X=right, Y=up, Z=out-of-screen
        # Blender camera: X=right, Y=up, Z=backward
        # Remap phone quaternion → Blender world space
        phone_q = Quaternion((qw, qx, qy, qz))

        # Rotate coordinate frame: phone Y-up → Blender Z-up
        # swap axes: Blender_x=phone_x, Blender_y=phone_z, Blender_z=-phone_y
        bq = Quaternion((phone_q.w, phone_q.x, phone_q.z, -phone_q.y))

        if props.use_smoothing:
            cur = obj.rotation_quaternion.copy()
            obj.rotation_quaternion = cur.slerp(bq, 1.0 - alpha)
        else:
            obj.rotation_quaternion = bq

        obj.rotation_mode = "QUATERNION"

    # ── Euler fallback ──
    elif "alpha" in data:
        # DeviceOrientation: alpha=yaw, beta=pitch, gamma=roll (degrees)
        yaw   = math.radians(data.get("alpha", 0))
        pitch = math.radians(data.get("beta",  0))
        roll  = math.radians(data.get("gamma", 0))

        # Convert to Blender Euler (ZYX)
        target = Euler((-pitch, roll, -yaw), "ZYX")

        if props.use_smoothing:
            cur = obj.rotation_euler
            factor = 1.0 - alpha
            obj.rotation_euler = Euler(
                (cur.x + (target.x - cur.x) * factor,
                 cur.y + (target.y - cur.y) * factor,
                 cur.z + (target.z - cur.z) * factor),
                "ZYX"
            )
        else:
            obj.rotation_euler = target
            obj.rotation_mode = "ZYX"

    # ── Tag viewport for redraw ──
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()

    return props.update_interval


_timer_registered = False


def register_timer():
    global _timer_registered
    if not _timer_registered:
        bpy.app.timers.register(_apply_rotation, persistent=True)
        _timer_registered = True


def unregister_timer():
    global _timer_registered
    if _timer_registered and bpy.app.timers.is_registered(_apply_rotation):
        bpy.app.timers.unregister(_apply_rotation)
    _timer_registered = False


# ──────────────────────────────────────────────
#  Properties
# ──────────────────────────────────────────────

class PhoneCamProps(bpy.types.PropertyGroup):
    port: bpy.props.IntProperty(
        name="Port", default=8765, min=1024, max=65535
    )
    camera_name: bpy.props.StringProperty(
        name="Camera", default="",
        description="Leave blank to use scene active camera"
    )
    update_interval: bpy.props.FloatProperty(
        name="Update Interval", default=0.016, min=0.008, max=0.5,
        description="Seconds between updates (~0.016 = 60fps)"
    )
    use_smoothing: bpy.props.BoolProperty(
        name="Smoothing", default=True
    )
    smoothing: bpy.props.FloatProperty(
        name="Smoothing Factor", default=0.85, min=0.0, max=0.99,
        description="0 = instant, 0.99 = very smooth"
    )
    is_running: bpy.props.BoolProperty(default=False)


# ──────────────────────────────────────────────
#  Operators
# ──────────────────────────────────────────────

class PHONECAM_OT_Start(bpy.types.Operator):
    bl_idname = "phonecam.start"
    bl_label = "Start Server"

    def execute(self, context):
        props = context.scene.phone_cam_props
        start_server(props.port)
        register_timer()
        props.is_running = True
        self.report({"INFO"}, f"Server started on port {props.port}")
        return {"FINISHED"}


class PHONECAM_OT_Stop(bpy.types.Operator):
    bl_idname = "phonecam.stop"
    bl_label = "Stop Server"

    def execute(self, context):
        props = context.scene.phone_cam_props
        stop_server()
        unregister_timer()
        props.is_running = False
        self.report({"INFO"}, "Server stopped")
        return {"FINISHED"}


class PHONECAM_OT_ResetRot(bpy.types.Operator):
    bl_idname = "phonecam.reset_rotation"
    bl_label = "Reset Camera Rotation"

    def execute(self, context):
        props = context.scene.phone_cam_props
        cam_name = props.camera_name
        obj = bpy.data.objects.get(cam_name) if cam_name else context.scene.camera
        if obj:
            obj.rotation_euler = (0, 0, 0)
            obj.rotation_mode = "XYZ"
        with _data_lock:
            _latest_data.clear()
        return {"FINISHED"}


# ──────────────────────────────────────────────
#  Panel
# ──────────────────────────────────────────────

class PHONECAM_PT_Panel(bpy.types.Panel):
    bl_label = "📱 Phone Camera Control"
    bl_idname = "PHONECAM_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phone Cam"

    def draw(self, context):
        layout = self.layout
        props = context.scene.phone_cam_props

        # Status indicator
        status_box = layout.box()
        row = status_box.row()
        if props.is_running:
            row.label(text="● LIVE", icon="RADIOBUT_ON")
            row.label(text=f"Port: {props.port}")
            row.label(text=f"Clients: {_client_count}")
        else:
            row.label(text="○ Offline", icon="RADIOBUT_OFF")

        layout.separator()

        # Server settings
        col = layout.column(align=True)
        col.label(text="Server Settings:", icon="NETWORK_DRIVE")
        col.prop(props, "port")
        col.prop_search(props, "camera_name", bpy.data, "objects",
                        text="Camera", icon="CAMERA_DATA")

        layout.separator()

        # Start/Stop
        row = layout.row(align=True)
        if not props.is_running:
            row.operator("phonecam.start", text="▶ Start", icon="PLAY")
        else:
            row.operator("phonecam.stop", text="■ Stop", icon="PAUSE")

        layout.separator()

        # Smoothing
        smooth_box = layout.box()
        smooth_box.label(text="Motion Smoothing:", icon="MOD_SMOOTH")
        smooth_box.prop(props, "use_smoothing")
        if props.use_smoothing:
            smooth_box.prop(props, "smoothing", slider=True)
        smooth_box.prop(props, "update_interval")

        layout.separator()
        layout.operator("phonecam.reset_rotation", text="↺ Reset Rotation", icon="LOOP_BACK")

        # IP help
        layout.separator()
        ip_box = layout.box()
        ip_box.label(text="Connect phone to:", icon="INFO")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "your-PC-IP"
        ip_box.label(text=f"ws://{ip}:{props.port}")


# ──────────────────────────────────────────────
#  Register
# ──────────────────────────────────────────────

classes = [
    PhoneCamProps,
    PHONECAM_OT_Start,
    PHONECAM_OT_Stop,
    PHONECAM_OT_ResetRot,
    PHONECAM_PT_Panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.phone_cam_props = bpy.props.PointerProperty(type=PhoneCamProps)


def unregister():
    stop_server()
    unregister_timer()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.phone_cam_props


if __name__ == "__main__":
    register()
