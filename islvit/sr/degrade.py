"""Image degradations: random ones to train the GAN, fixed ones to test it.

Training uses a simplified version of Real-ESRGAN's second-order degradation:
blur, resample, noise and JPEG, applied twice, then a final downscale to the
low-resolution size. Real low-resolution video is not a clean bicubic downscale
-- it is a blurry, noisy, compressed one -- and a generator trained only on clean
downscales sharpens the compression artefacts instead of removing them.

The study uses one fixed degradation per level (``degrade_frame``): an area
downscale to a target height followed by a single JPEG encode. It is
deterministic, so every arm of the comparison sees byte-identical inputs.
"""

from __future__ import annotations

import cv2
import numpy as np

INTERPOLATIONS = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)


def parse_spec(spec: str | None) -> tuple[int, int | None] | None:
    """'360:60' -> (360, 60); '240' -> (240, None); None/'' -> None."""
    if not spec:
        return None
    height, _, quality = str(spec).partition(":")
    return int(height), (int(quality) if quality else None)


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def degrade_frame(frame: np.ndarray, height: int, quality: int | None = None) -> np.ndarray:
    """Deterministic study degradation of one BGR frame: area-downscale to
    ``height`` (never upscale), then optionally one JPEG encode."""
    h, w = frame.shape[:2]
    if height < h:
        frame = cv2.resize(frame, (max(1, int(round(w * height / h))), height), interpolation=cv2.INTER_AREA)
    if quality:
        frame = jpeg(frame, quality)
    return frame


# --------------------------------------------------------------------------- #
# Random degradation for training
# --------------------------------------------------------------------------- #
def _gaussian_blur(img: np.ndarray, rng: np.random.Generator, sigma: tuple[float, float]) -> np.ndarray:
    s = rng.uniform(*sigma)
    k = int(2 * round(3 * s) + 1)
    return cv2.GaussianBlur(img, (k, k), s)


def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Signing hands move fast at 25 fps, so the blur that matters most is directional.
    length = int(rng.integers(3, 10))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _blur(img, rng, sigma):
    if rng.random() < 0.25:
        return _motion_blur(img, rng)
    return _gaussian_blur(img, rng, sigma)


def _resize(img: np.ndarray, rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    factor = rng.uniform(low, high)
    h, w = img.shape[:2]
    size = (max(4, int(round(w * factor))), max(4, int(round(h * factor))))
    return cv2.resize(img, size, interpolation=INTERPOLATIONS[int(rng.integers(len(INTERPOLATIONS)))])


def _noise(img: np.ndarray, rng: np.random.Generator, sigma: tuple[float, float]) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.random() < 0.6:
        s = rng.uniform(*sigma)
        if rng.random() < 0.4:  # colourless sensor noise
            out += rng.normal(0, s, out.shape[:2])[..., None]
        else:
            out += rng.normal(0, s, out.shape)
    else:  # shot noise, scaled the way low light behaves
        scale = 10 ** rng.uniform(1.6, 3.0)
        out = rng.poisson(np.clip(out, 0, 255) / 255 * scale) / scale * 255
    return np.clip(out, 0, 255).astype(np.uint8)


# H.264 intra coding, the compression real phone and glasses video carries (JPEG is a
# stand-in for it). One persistent intra-only encoder/decoder pair per (size, CRF) and
# per process: building an x264 encoder costs ~1 s on this machine, encoding a 32 px
# patch with a warm one ~3 ms.
H264_CRF = (23, 28, 33, 38, 43)
_CODECS: dict = {}


def h264(image: np.ndarray, crf: int) -> np.ndarray:
    """One BGR uint8 image through an H.264 intra encode and decode (4:2:0, like a camera)."""
    import av
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE) if (h % 2 or w % 2) else image
    key = (padded.shape[0], padded.shape[1], crf)
    if key not in _CODECS:
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width, encoder.height, encoder.pix_fmt = padded.shape[1], padded.shape[0], "yuv420p"
        encoder.options = {"crf": str(crf), "preset": "veryfast", "tune": "zerolatency", "threads": "1",
                           "x264-params": "keyint=1:min-keyint=1:scenecut=0"}
        encoder.open()
        _CODECS[key] = (encoder, av.CodecContext.create("h264", "r"))
    encoder, decoder = _CODECS[key]
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(padded[..., ::-1]), format="rgb24").reformat(format="yuv420p")
    decoded = [f for packet in encoder.encode(frame) for f in decoder.decode(packet)]
    if len(decoded) != 1:           # a codec that buffers would hand back the wrong frame: refuse it
        raise RuntimeError(f"h264 round trip returned {len(decoded)} frames")
    return np.ascontiguousarray(decoded[0].to_ndarray(format="rgb24")[:h, :w, ::-1])


def random_degrade(hr: np.ndarray, rng: np.random.Generator, scale: int = 4, codec_prob: float = 0.0) -> np.ndarray:
    """HR uint8 image (H, W, 3) -> LR uint8 image (H/scale, W/scale, 3)."""
    h, w = hr.shape[:2]
    img = hr
    # First order.
    if rng.random() < 0.9:
        img = _blur(img, rng, (0.2, 2.4))
    img = _resize(img, rng, 0.4, 1.3)
    img = _noise(img, rng, (1.0, 18.0))
    img = jpeg(img, int(rng.integers(40, 96)))
    # Second order, milder.
    if rng.random() < 0.8:
        if rng.random() < 0.8:
            img = _blur(img, rng, (0.2, 1.4))
        img = _resize(img, rng, 0.6, 1.2)
        img = _noise(img, rng, (1.0, 12.0))
    # Final size, then compression in either order (a camera compresses after
    # scaling; a re-shared clip is scaled after compression).
    size = (w // scale, h // scale)
    interpolation = INTERPOLATIONS[int(rng.integers(len(INTERPOLATIONS)))]
    if codec_prob and rng.random() < codec_prob:
        # A camera scales, then encodes with a video codec.
        img = h264(cv2.resize(img, size, interpolation=interpolation), int(rng.choice(H264_CRF)))
    elif rng.random() < 0.5:
        img = jpeg(cv2.resize(img, size, interpolation=interpolation), int(rng.integers(40, 96)))
    else:
        img = cv2.resize(jpeg(img, int(rng.integers(40, 96))), size, interpolation=interpolation)
    return img
