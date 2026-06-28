"""LeRobot dataset frame loader — extracts video frames as Gemma 4 data URIs.

Usage:
    source = LeRobotFrameSource("lerobot/pusht")
    print(source.info())
    frame = source.get_frame(global_idx=0)
    frame = source.get_frame(episode=0, frame_idx=0)
    for f in source.iter_episode(0, step=5):
        ...  # f.image_uri, f.action, f.state
"""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from lerobot.datasets import LeRobotDataset


class VideoFrame:
    """A single decoded video frame from a LeRobot dataset, ready for Gemma 4."""

    __slots__ = (
        "episode_index",
        "frame_index",
        "image_uri",
        "image_pil",
        "image_size",
        "action",
        "state",
        "timestamp",
        "camera_key",
    )

    def __init__(
        self,
        episode_index: int,
        frame_index: int,
        image_uri: str,
        image_pil: Image.Image,
        action: list[float],
        state: list[float],
        timestamp: float,
        camera_key: str,
    ) -> None:
        self.episode_index = episode_index
        self.frame_index = frame_index
        self.image_uri = image_uri
        self.image_pil = image_pil
        self.image_size = (image_pil.width, image_pil.height)
        self.action = action
        self.state = state
        self.timestamp = timestamp
        self.camera_key = camera_key

    def __repr__(self) -> str:
        return (
            f"VideoFrame(ep={self.episode_index}, frame={self.frame_index}, "
            f"cam={self.camera_key}, img={self.image_size}, "
            f"act={_fmt_float_list(self.action)})"
        )


def _fmt_float_list(vals: list[float], decimals: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in vals[:4]) + ("..." if len(vals) > 4 else "") + "]"


def _tensor_to_uri(tensor: torch.Tensor, quality: int = 90) -> tuple[str, Image.Image]:
    """Convert a CHW float32/uint8 tensor to a JPEG data URI + PIL image.

    Handles float tensors in [0,1] as well as uint8 tensors in [0,255].
    """
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        arr = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    else:
        arr = tensor.permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", pil_img


class LeRobotFrameSource:
    """Wraps a LeRobotDataset and provides frame access for Gemma 4 multimodal."""

    def __init__(self, repo_id: str, camera_key: str | None = None) -> None:
        self._dataset = LeRobotDataset(repo_id)
        self._camera_key = camera_key or self._dataset.meta.camera_keys[0]
        # Build episode offset index (frame index where each episode starts)
        self._episode_offsets: list[int] = self._build_episode_index()

    # -- public properties -------------------------------------------------

    @property
    def repo_id(self) -> str:
        return self._dataset.repo_id

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def fps(self) -> int:
        return self._dataset.fps

    @property
    def camera_key(self) -> str:
        return self._camera_key

    @property
    def camera_keys(self) -> list[str]:
        return self._dataset.meta.camera_keys

    # -- episode index -----------------------------------------------------

    def _build_episode_index(self) -> list[int]:
        """Scan to find the global frame index where each episode starts."""
        offsets: list[int] = [0]
        prev_ep = 0
        stride = min(50, max(1, self._dataset.num_frames // 200))
        for i in range(stride, self._dataset.num_frames, stride):
            ep = int(self._dataset[i]["episode_index"])
            if ep != prev_ep:
                lo, hi = i - stride, i
                while lo < hi:
                    mid = (lo + hi) // 2
                    if int(self._dataset[mid]["episode_index"]) == prev_ep:
                        lo = mid + 1
                    else:
                        hi = mid
                offsets.append(lo)
                prev_ep = int(self._dataset[lo]["episode_index"])
        return offsets

    def episode_range(self, ep_idx: int) -> tuple[int, int]:
        """(start_frame, end_frame_exclusive) for episode *ep_idx*."""
        if ep_idx < 0 or ep_idx >= self.num_episodes:
            msg = f"Episode {ep_idx} out of range [0, {self.num_episodes})"
            raise IndexError(msg)
        from_idx = self._episode_offsets[ep_idx]
        if ep_idx + 1 < len(self._episode_offsets):
            to_idx = self._episode_offsets[ep_idx + 1]
        else:
            # Scan forward for the next episode boundary
            to_idx = self._dataset.num_frames
            for i in range(from_idx + 1, self._dataset.num_frames):
                if int(self._dataset[i]["episode_index"]) != ep_idx:
                    to_idx = i
                    break
        return from_idx, to_idx

    def episode_frames(self, ep_idx: int) -> int:
        f, t = self.episode_range(ep_idx)
        return t - f

    # -- frame access ------------------------------------------------------

    def get_frame(
        self,
        global_idx: int | None = None,
        *,
        episode: int | None = None,
        frame_idx: int | None = None,
    ) -> VideoFrame:
        """Get a single frame by global index or by ``(episode, frame_idx)``."""
        if global_idx is not None:
            idx = global_idx
        elif episode is not None:
            from_idx, _ = self.episode_range(episode)
            idx = from_idx + (frame_idx or 0)
        else:
            msg = "Provide global_idx or (episode + frame_idx)"
            raise ValueError(msg)

        raw = self._dataset[idx]
        img_tensor: torch.Tensor = raw[self._camera_key]
        uri, pil_img = _tensor_to_uri(img_tensor)
        return VideoFrame(
            episode_index=int(raw["episode_index"]),
            frame_index=int(raw["frame_index"]),
            image_uri=uri,
            image_pil=pil_img,
            action=raw["action"].tolist() if isinstance(raw["action"], torch.Tensor) else list(raw["action"]),
            state=raw["observation.state"].tolist()
            if isinstance(raw["observation.state"], torch.Tensor)
            else list(raw["observation.state"]),
            timestamp=float(raw.get("timestamp", 0.0)),
            camera_key=self._camera_key,
        )

    def iter_episode(self, ep_idx: int, step: int = 1):
        """Yield :class:`VideoFrame` objects from an episode, striding by ``step``."""
        from_idx, to_idx = self.episode_range(ep_idx)
        for i in range(from_idx, to_idx, step):
            yield self.get_frame(global_idx=i)

    # -- info --------------------------------------------------------------

    def info(self) -> str:
        ep0_frames = self.episode_frames(0)
        cams = ", ".join(self.camera_keys)
        return (
            f"LeRobot dataset: {self.repo_id}\n"
            f"  Episodes: {self.num_episodes}\n"
            f"  Frames:   {self.num_frames}\n"
            f"  FPS:      {self.fps}\n"
            f"  Cameras:  {cams}\n"
            f"  Primary:  {self.camera_key}\n"
            f"  Ep 0:     {ep0_frames} frames"
        )
