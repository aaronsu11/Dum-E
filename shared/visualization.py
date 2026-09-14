"""Optional camera preview; does not access robot hardware."""
import numpy as np
import matplotlib.pyplot as plt

# Global figure and axis for continuous streaming
_fig = None
_ax = None


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env.
    Continuously streams images in the same window without creating new ones.
    """
    global _fig, _ax

    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    # Calculate new dimensions while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = max(720 / w, 720 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Create figure and axis only once
    if _fig is None or _ax is None:
        _fig, _ax = plt.subplots(figsize=(new_w / 100, new_h / 100))
        _ax.set_title("Camera View")
        _ax.axis("off")
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    # Clear previous image and display new one
    _ax.clear()
    _ax.imshow(img)
    _ax.set_title("Camera View")
    _ax.axis("off")

    # Update the display
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)  # Small pause to allow GUI to update
