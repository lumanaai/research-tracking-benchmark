"""
End-to-end test that verifies the local_vcc Docker container can handle:
  1. The longest prompt from the prompt_map LUT
  2. A full 1280×720 image
  3. Images at much higher resolution (e.g. 4K+ camera stills), which are
     NOT resized anywhere before reaching the VLM (see app/image_utils.py)

This validates that ctx-size (4096) is sufficient for production workloads,
and that --image-max-tokens actually bounds vision-token usage regardless of
input resolution: llama.cpp's vision encoder uses *dynamic* resolution, so
without an explicit cap, prompt_tokens scales with image pixel count and a
high-res image can blow the context budget outright (measured empirically:
a 2560x1440 image alone used ~3800/4096 tokens, and 3840x2160 failed with a
"request exceeds the available context size" 400 error against ctx-size
4096 and no --image-max-tokens).

Note: --image-min-tokens is deliberately NOT set (removed after confirming
it only forces small images to be upscaled - pure wasted compute, no
accuracy benefit, since the model's dynamic-resolution training already
covers small native token counts). --image-max-tokens alone is sufficient
to bound the worst case; re-verified empirically across every prompt_map
entry x every resolution below with --image-max-tokens=2048: worst case
was 2254/4096 tokens (55%), comfortable headroom under ctx-size.

Requirements:
  - The lumixai/local_vcc:latest image must be built.
  - An NVIDIA GPU must be available.
  - Docker must be accessible.

Run:
    pytest analyzer_manager/local_vcc/tests/test_e2e_vlm_capacity.py -v -s
"""

import base64
import io
import json
import subprocess
import time
import sys
import urllib.request

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONTAINER_NAME = "local_vcc_e2e_test"
IMAGE_NAME = "lumixai/local_vcc:latest"
PORT = 7998  # use a different port to avoid conflicts with running services
HEALTH_TIMEOUT = 300  # seconds
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CTX_SIZE = 4096  # must match LLAMA_CTX_SIZE default in entrypoint.sh
IMAGE_MAX_TOKENS = 2048  # must match LLAMA_IMAGE_MAX_TOKENS default in entrypoint.sh

# ---------------------------------------------------------------------------
# The longest prompt from the prompt_map LUT (copy to avoid import issues
# when running outside the container)
# ---------------------------------------------------------------------------
LONGEST_PROMPTS = {
    "BRANDISHING_WEAPON": """
        You are tasked with identify if the person in the image is carrying a firearm in his hands.
        First, describe what the person is holding in his hands. Notice similar objects like phones, keys, tablets, hand drills, bottles, or cans. 
        Also notice what the person is carrying on his belt or over his shoulder.
        The description should be short and concise - no more than 20 words.
        Then answer:
        visible_firearm - Is there a visible firearm (such as a gun or similar weapon) in this image? 
        firearm_in_hands - Is the firearm actively held in the person's hand(s).
        """,
    "FIRE": """
        You are tasked with determining whether there is fire in the image.

        1. **Image Description**:
        - Describe the image any elements related to fire (e.g., flames, smoke).
        - Explicitly mention features that resemble fire but are not (e.g., lights, reflections, haze, or corrupted pixels).
        - Prioritize evidence of actual fire behavior (e.g., flickering flames, rising smoke) over ambiguous features.
        - The description should be short and concise - no more than 20 words.

        3. **Final Answer**:
        - Based on your analysis, state if there is fire in the image.

        Note: Be cautious of light sources, reflections, and other false-positive indicators. Explain your reasoning clearly to avoid misinterpretation.
        """,
    "VIOLENCE": """
        These are frames from a video arranged in a collage.
        Your task is to determine whether there is physical violence in this video.
        **Definition**: Any physical aggression must be classified as violence, even if it could be playful, or staged. Do not infer intent or context. 
        If physical aggression is visible, it counts as violence.
        First, describe the physical interactions and elements that indicate violence.
        Then, describe any factors that could lead to misinterpretation.
        Keep your description short and concise—no more than 30 words.
        Finally, answer: Is there physical violence?
    """,
}

# Pick the longest prompt by character count
LONGEST_PROMPT = max(LONGEST_PROMPTS.values(), key=len)


def _generate_test_image_b64(width: int, height: int) -> str:
    """Generate a realistic-sized RGB image and return as base64 JPEG data URL."""
    from PIL import Image, ImageDraw
    import numpy as np

    # Create a non-trivial image (gradient + shapes) to exercise the vision encoder
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    # Draw some shapes to make it less compressible
    draw.rectangle([width // 10, height // 10, width // 3, height // 3], fill="red")
    draw.ellipse([width // 2, height // 3, width * 3 // 4, height * 2 // 3], fill="blue")
    draw.text((width // 6, height // 15), f"TEST IMAGE {width}x{height}", fill="white")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _docker_run():
    """Start the local_vcc container with llama-server only (skip worker)."""
    # Remove any leftover container
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
    )
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "-p", f"{PORT}:7999",
        "--entrypoint", "/app/llama-server",
        IMAGE_NAME,
        "--host", "0.0.0.0",
        "--port", "7999",
        "--model", "/models/model.gguf",
        "--mmproj", "/models/mmproj.gguf",
        "--alias", "local_vcc",
        "--n-gpu-layers", "-1",
        "--jinja",
        "--flash-attn", "on",
        "--ctx-size", str(CTX_SIZE),
        "--parallel", "1",
        "--image-max-tokens", str(IMAGE_MAX_TOKENS),
        "--batch-size", "4096",
        "--ubatch-size", "512",
        "--split-mode", "none",
        "--main-gpu", "0",
        "-lv", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to start container: {result.stderr}")
    return result.stdout.strip()


def _wait_healthy(timeout: int = HEALTH_TIMEOUT):
    """Wait for the llama-server /health endpoint to return ok."""
    url = f"http://localhost:{PORT}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    # On timeout, dump container logs
    logs = subprocess.run(
        ["docker", "logs", "--tail", "80", CONTAINER_NAME],
        capture_output=True, text=True,
    )
    pytest.fail(
        f"llama-server did not become healthy within {timeout}s.\n"
        f"Container logs:\n{logs.stdout}\n{logs.stderr}"
    )


def _docker_cleanup():
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)


def _send_chat_completion(image_url: str, prompt: str, max_tokens: int = 100) -> dict:
    """Send a /v1/chat/completions request and return the parsed response."""
    payload = {
        "model": "local_vcc",
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    url = f"http://localhost:{PORT}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def vlm_container():
    """Start the VLM container for the test module; tear down after."""
    _docker_run()
    try:
        _wait_healthy()
        yield
    finally:
        _docker_cleanup()


class TestVlmCapacity:
    """Validate that the server handles max prompt + max image within ctx budget."""

    def test_longest_prompt_with_full_image(self):
        """Send the longest LUT prompt + 1280x720 image.

        Asserts:
        - Server returns HTTP 200 (no ctx overflow error)
        - Response contains generated text (non-empty)
        - prompt_tokens are within ctx-size budget
        """
        image_url = _generate_test_image_b64(IMAGE_WIDTH, IMAGE_HEIGHT)
        result = _send_chat_completion(image_url, LONGEST_PROMPT, max_tokens=100)

        # Validate structure
        assert "choices" in result, f"Unexpected response: {result}"
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert content and len(content.strip()) > 0, "Empty VLM response"

        # Validate token usage is within budget
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        print(f"\n  prompt_tokens={prompt_tokens}, total_tokens={total_tokens}")
        print(f"  response: {content[:200]}")

        # Must not exceed ctx-size
        assert total_tokens <= CTX_SIZE, (
            f"total_tokens ({total_tokens}) exceeds ctx-size {CTX_SIZE}!"
        )
        # Sanity: the image should be contributing a substantial share of the
        # prompt (native 1280x720 alone measured ~922 vision tokens with no
        # --image-min-tokens floor in place). Not tied to any forced minimum.
        assert prompt_tokens >= 800, (
            f"Expected the image to contribute a substantial share of prompt_tokens, got {prompt_tokens}"
        )

    def test_all_lut_prompts_with_full_image(self):
        """Run every prompt from the LUT with a 1280x720 image.

        This ensures no prompt causes a context overflow.
        """
        image_url = _generate_test_image_b64(IMAGE_WIDTH, IMAGE_HEIGHT)
        for name, prompt in LONGEST_PROMPTS.items():
            print(f"\n  Testing prompt: {name} ({len(prompt)} chars)")
            result = _send_chat_completion(image_url, prompt, max_tokens=80)

            assert "choices" in result, f"Failed for {name}: {result}"
            content = result["choices"][0]["message"]["content"]
            assert content and len(content.strip()) > 0, f"Empty response for {name}"

            usage = result.get("usage", {})
            total = usage.get("total_tokens", 0)
            prompt_tok = usage.get("prompt_tokens", 0)
            print(f"    prompt_tokens={prompt_tok}, total_tokens={total}")
            assert total <= CTX_SIZE, (
                f"{name}: total_tokens ({total}) exceeds ctx-size!"
            )

    @pytest.mark.parametrize("width,height", [
        (1920, 1080),  # Full HD
        (2560, 1440),  # QHD - without --image-max-tokens this alone used ~3800/4096
        (3840, 2160),  # 4K - without --image-max-tokens this failed outright (400)
        (4656, 3492),  # ~16MP, a common IP-camera still resolution
    ])
    def test_high_resolution_image_does_not_exceed_context(self, width, height):
        """Images are never resized before reaching the VLM (see
        app/image_utils.py) - a camera alert can plausibly be 4K+. Without
        --image-max-tokens, vision-token usage scales with resolution and
        these exact sizes either ate most of the ctx-size budget or exceeded
        it outright. --image-max-tokens pins vision tokens to a fixed,
        resolution-independent budget, so this must keep succeeding
        regardless of how large the input image is.
        """
        image_url = _generate_test_image_b64(width, height)
        result = _send_chat_completion(image_url, LONGEST_PROMPT, max_tokens=80)

        assert "choices" in result, f"{width}x{height}: unexpected response: {result}"
        content = result["choices"][0]["message"]["content"]
        assert content and len(content.strip()) > 0, f"{width}x{height}: empty VLM response"

        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        print(f"\n  {width}x{height}: prompt_tokens={prompt_tokens}, total_tokens={total_tokens}")

        assert total_tokens <= CTX_SIZE, (
            f"{width}x{height}: total_tokens ({total_tokens}) exceeds ctx-size {CTX_SIZE}! "
            f"(--image-max-tokens should have capped this regardless of resolution)"
        )
        # With --image-max-tokens set, resolution shouldn't meaningfully move prompt_tokens
        # beyond the cap plus a reasonable text/prompt allowance.
        assert prompt_tokens <= IMAGE_MAX_TOKENS + 500, (
            f"{width}x{height}: prompt_tokens ({prompt_tokens}) is much higher than expected "
            f"for a capped image budget of {IMAGE_MAX_TOKENS} - is --image-max-tokens actually set?"
        )

    def test_health_endpoint(self):
        """Sanity: /health returns ok."""
        url = f"http://localhost:{PORT}/health"
        resp = urllib.request.urlopen(url, timeout=10)
        data = json.loads(resp.read())
        assert data["status"] == "ok"

