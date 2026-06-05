import numpy as np
import base64
from PIL import Image
import io

def obs_to_dict(obs: dict) -> dict:
    # MCP requires JSON-serialisable types; convert numpy arrays to plain lists
    obs_c = obs.copy()
    for k, v in obs_c.items():
        if isinstance(v, np.ndarray):
            obs_c[k] = v.tolist()
    return obs_c

def frame_to_base64(frame: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    # Encode a rendered RGB frame as a base64 string for transmission over MCP
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality if format == "JPEG" else None)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")