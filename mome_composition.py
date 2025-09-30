import hub_utils 
import momeutils 
import re
import networkx as nx
import sys 
import json 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import glob
import random
import hashlib
import pickle

def _hash_content(content) -> str:
    """Return a stable hash for node_content (using pickle + sha256)."""
    try:
        data = pickle.dumps(content)
    except Exception:
        # fallback: str representation
        data = str(content).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:50]


def get_random_font(base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.expanduser('~'), "Documents", "Fonts")
    files = glob.glob(os.path.join(base_path, '**', '*.ttf'), recursive=True)
    files += glob.glob(os.path.join(base_path, '**', '*.otf'), recursive=True)
    if not files:
        raise FileNotFoundError("No font files found in the specified directory.")
    return random.choice(files)

def load_image_stage(stage, current_image):
    """Handles 'image' stage: loads or replaces the base image."""
    path = stage['path']
    return Image.open(path).convert("RGBA")

def get_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)

def create_gradient(size, color1, color2, horizontal=True):
    width, height = size
    gradient = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(gradient)
    for i in range(width if horizontal else height):
        ratio = i / (width - 1) if horizontal else i / (height - 1)
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        a = int(color1[3] * (1 - ratio) + color2[3] * ratio)
        if horizontal:
            draw.line([(i, 0), (i, height)], fill=(r, g, b, a))
        else:
            draw.line([(0, i), (width, i)], fill=(r, g, b, a))
    return gradient

def create_split_color_bars(im_size, bars_height, bars_top, text_x, text_width, horizontal=True):
    """
    Draws colored bars above/below or left/right of a text area, avoiding the text region.
    Ensures coordinates are within image bounds and x1 >= x0.
    """
    width, height = im_size
    bars_img = Image.new("RGBA", (width, bars_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bars_img)
    colors = [
        (255, 0, 0, 255),      # Red
        (255, 165, 0, 255),    # Orange
        (255, 255, 0, 255),    # Yellow
        (0, 255, 0, 255),      # Green
        (0, 127, 255, 255),    # Blue
        (75, 0, 130, 255),     # Indigo
        (148, 0, 211, 255),    # Violet
    ]
    n = len(colors)
    if horizontal:
        bar_height = bars_height // n
        for i, color in enumerate(colors):
            y0 = i * bar_height
            y1 = (i + 1) * bar_height if i < n - 1 else bars_height
            # Left bar
            x0_left = 0
            x1_left = max(0, min(text_x, width))
            if x1_left > x0_left:
                draw.rectangle([x0_left, y0, x1_left, y1], fill=color)
            # Right bar
            x0_right = max(0, min(text_x + text_width, width))
            x1_right = width
            if x1_right > x0_right:
                draw.rectangle([x0_right, y0, x1_right, y1], fill=color)
    else:
        bar_width = width // n
        for i, color in enumerate(colors):
            x0 = i * bar_width
            x1 = (i + 1) * bar_width if i < n - 1 else width
            y0_top = 0
            y1_top = max(0, min(text_x, bars_height))
            if y1_top > y0_top:
                draw.rectangle([x0, y0_top, x1, y1_top], fill=color)
            y0_bottom = max(0, min(text_x + text_width, bars_height))
            y1_bottom = bars_height
            if y1_bottom > y0_bottom:
                draw.rectangle([x0, y0_bottom, x1, y1_bottom], fill=color)
    full_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    full_img.paste(bars_img, (0, bars_top), mask=bars_img)
    return full_img


def create_text_gradient(text, font, text_width, text_height, color1, color2, horizontal=True):
    gradient = create_gradient((text_width, text_height), color1, color2, horizontal=horizontal)
    text_mask = Image.new("L", (text_width, text_height), 0)
    draw_text_mask = ImageDraw.Draw(text_mask)
    draw_text_mask.text((0, 0), text, font=font, fill=255)
    text_gradient = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    text_gradient.paste(gradient, (0, 0), mask=text_mask)
    return text_gradient, text_mask


def add_text(params, current_inputs):
    """
    Adds text to the image at a specified location or centered by default.

    Args:
        params (dict): Dictionary with keys:
            - text (str)
            - font_size (int)
            - color1, color2 (for gradient)
            - horizontal (bool)
            - x, y (optional): explicit position (int for pixels, float for fraction)
            - centered (bool, optional): if True, center horizontally and vertically (default)
            - center_x (bool, optional): if True, center horizontally
            - center_y (bool, optional): if True, center vertically
            - bars (bool, optional): if True, add bars behind text
            - bars_height_ratio (float, optional)
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data

    Returns:
        PIL.Image.Image: The image with text added.
        dict: Metadata about the text placement.
    """

    current_image = current_inputs[0]['image']
    im_size = current_image.size

    text = params['text']
    font_size = params.get('font_size', 100)
    gradient = params.get("gradient", False)

    if gradient:
        color1 = params.get('color1', (255, 0, 0, 255))
        color2 = params.get('color2', (0, 0, 255, 255))
    else:
        color1 = color2 = params.get("color", (0, 0, 0, 255))

    horizontal = params.get('horizontal', True)
    font_path = params.get('font_path', get_random_font())
    font = get_font(font_path, font_size)

    # Calculate text size
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    img_width, img_height = im_size

    # Determine position
    x = params.get('x')
    y = params.get('y')
    centered = params.get('centered', True)  # Default is centered
    center_x = params.get('center_x', False)
    center_y = params.get('center_y', False)

    # Convert fractional x/y to pixel coordinates if needed
    if isinstance(x, float) and 0.0 <= x <= 1.0:
        x = int(x * (img_width - text_width))
    if isinstance(y, float) and 0.0 <= y <= 1.0:
        y = int(y * (img_height - text_height))

    if x is not None and y is not None:
        pass  # Use explicit coordinates (already converted if fractional)
    else:
        if centered or (center_x and center_y):
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
        else:
            if center_x or x is None:
                x = (img_width - text_width) // 2
            if center_y or y is None:
                y = (img_height - text_height) // 2

    # Optionally add bars before text
    if params.get('bars', False):
        bars_height_ratio = params.get('bars_height_ratio', 0.9)
        bars_height = int(text_height * bars_height_ratio)
        bars_top = y - int(text_height * 0.05)
        bars_layer = create_split_color_bars(
            im_size, bars_height, bars_top, x, text_width, horizontal=horizontal
        )
        current_image = Image.alpha_composite(current_image, bars_layer)

    text_gradient, text_mask = create_text_gradient(
        text, font, text_width, text_height, color1, color2, horizontal=horizontal
    )
    current_image.paste(text_gradient, (x, y), mask=text_mask)

    return current_image, {
        'x': x,
        'y': y,
        'text_width': text_width,
        'text_height': text_height
    }


def bars_stage(stage, current_image, last_text_info):
    """
    Handles 'bars' stage. Returns updated image.
    Expects:
        - bars_height
        - horizontal
        - text_x, text_width, bars_top
    """
    im_size = current_image.size
    # Use last_text_info for bars position if not explicitly given
    text_x = stage.get('text_x', last_text_info['x'])
    text_width = stage.get('text_width', last_text_info['text_width'])
    bars_top = stage.get('bars_top', last_text_info['y'] + int(last_text_info['text_height'] * 0.5))
    bars_height = stage.get('bars_height', int(last_text_info['text_height'] * stage.get('bars_height_ratio', 0.4)))
    horizontal = stage.get('horizontal', True)
    bars_layer = create_split_color_bars(
        im_size, bars_height, bars_top, text_x, text_width, horizontal=horizontal
    )
    result = Image.alpha_composite(current_image, bars_layer)
    return result


def vhs_scanline(params, current_inputs):
    """
    Adds VHS-style horizontal scanlines to the input image.

    Args:
        params (dict): Dictionary with optional keys:
            - 'line_color' (tuple): RGB color of the scanlines (default: black).
            - 'line_opacity' (int): Opacity of the scanlines (0-255, default: 64).
            - 'line_spacing' (int): Number of pixels between scanlines (default: 3).
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data

    Returns:
        PIL.Image.Image: The image with scanlines applied.
        dict: Metadata with used parameters.
    """
    line_color = params.get('line_color', (0, 0, 0))
    line_opacity = params.get('line_opacity', 64)
    line_spacing = params.get('line_spacing', 3)

    # Extract the image from the first input
    current_image = current_inputs[0]['image']

    # Ensure image is in RGBA mode for transparency
    img = current_image.convert("RGBA")
    width, height = img.size

    # Create a transparent overlay for scanlines
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw horizontal lines
    for y in range(0, height, line_spacing):
        draw.line(
            [(0, y), (width, y)],
            fill=line_color + (line_opacity,),
            width=1
        )

    # Composite the overlay onto the original image
    result = Image.alpha_composite(img, overlay)

    # Return the result and metadata
    return result, {
        "line_color": line_color,
        "line_opacity": line_opacity,
        "line_spacing": line_spacing
    }

def vhs_grain(params, current_inputs):
    """
    Adds VHS-style noise/grain to the input image.

    Args:
        params (dict): Dictionary with optional keys:
            - 'intensity' (float): Noise intensity (0.0 to 1.0, default: 0.2).
            - 'colored' (bool): If True, use colored noise; otherwise, grayscale noise.
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data

    Returns:
        (PIL.Image.Image, dict): The image with noise/grain applied, and metadata.
    """
    intensity = params.get("intensity", 0.2)
    colored = params.get("colored", True)

    # Extract the image from the first input
    current_image = current_inputs[0]['image']

    # Convert image to numpy array
    img = np.array(current_image).astype(np.float32)
    h, w = img.shape[:2]

    # Generate noise
    if colored and img.ndim == 3:
        noise = np.random.randn(h, w, img.shape[-1]) * 255 * intensity
    else:
        noise = np.random.randn(h, w, 1) * 255 * intensity
        if img.ndim == 3:
            noise = np.repeat(noise, img.shape[-1], axis=2)

    # Add noise and clip to valid range
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(noisy_img), {"intensity": intensity, "colored": colored}



def soft_focus_blur(params, current_inputs):
    """
    Applies a soft focus and blur effect to the input image.
    
    Args:
        params (dict): Dictionary with optional keys:
            - 'blur_radius' (float): The radius for Gaussian blur. Default is 4.0.
            - 'blend_alpha' (float): Blend strength between original and blurred image (0-1). Default is 0.6.
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data
        
    Returns:
        PIL.Image.Image: The image with soft focus and blur effect applied.
    """

    blur_radius = params.get('blur_radius', 4.0)
    blend_alpha = params.get('blend_alpha', 0.6)
    
    current_image = current_inputs[0]['image']
    # Apply Gaussian blur
    blurred = current_image.filter(ImageFilter.GaussianBlur(blur_radius))

    
    # Blend the original and blurred images
    soft_focus = Image.blend(current_image, blurred, blend_alpha)
    
    return soft_focus, {"blur_radius": blur_radius, "blend_alpha": blend_alpha}


def soft_focus_blur_outside_rect(params, current_inputs):
    """
    Applies a soft focus and blur effect to the input image, but only outside a specified rectangle.
    Optionally draws a border around the rectangle. Rectangle is specified as (x, y, w, h), where
    values can be absolute (int) or relative (float in [0,1]). Accepts rect as list or tuple.

    Args:
        params (dict): Dictionary with optional keys:
            - 'blur_radius' (float): The radius for Gaussian blur. Default is 4.0.
            - 'blend_alpha' (float): Blend strength between original and blurred image (0-1). Default is 0.6.
            - 'rect' (list or tuple): [x, y, w, h] of rectangle. If any value is float, it's relative.
            - 'rect_border_width' (int): Width of the border in pixels. Default is 0 (no border).
            - 'rect_border_color' (tuple): RGB color of the border. Default is (255, 255, 255).
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data

    Returns:
        PIL.Image.Image: The image with soft focus and blur effect applied outside the rectangle, with optional border.
        dict: Parameters used.
    """
    blur_radius = params.get('blur_radius', 4.0)
    blend_alpha = params.get('blend_alpha', 0.6)
    rect_border_width = params.get('rect_border_width', 0)
    rect_border_color = params.get('rect_border_color', (255, 255, 255))
    current_image = current_inputs[0]['image']

    w_img, h_img = current_image.size
    # Default rectangle: center 50% of image
    default_rect = [w_img // 4, h_img // 4, w_img // 2, h_img // 2]
    rect = params.get('rect', default_rect)
    rect = list(rect)  # Ensure it's a list for mutability

    # Convert relative rect coordinates to absolute if needed
    def to_abs(val, maxval):
        if isinstance(val, float) and 0.0 <= val <= 1.0:
            return int(round(val * maxval))
        return int(val)

    if any(isinstance(v, float) and 0.0 <= v <= 1.0 for v in rect):
        x = to_abs(rect[0], w_img)
        y = to_abs(rect[1], h_img)
        w = to_abs(rect[2], w_img)
        h = to_abs(rect[3], h_img)
    else:
        x, y, w, h = (int(v) for v in rect)

    # Clamp rectangle to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    abs_rect = (x1, y1, x2, y2)

    # Apply Gaussian blur to the whole image
    blurred = current_image.filter(ImageFilter.GaussianBlur(blur_radius))
    # Blend the original and blurred images
    soft_focus = Image.blend(current_image, blurred, blend_alpha)

    # Create a mask: white inside rect, black outside
    mask = Image.new("L", current_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(abs_rect, fill=255)

    # Composite: inside rect is original, outside is soft_focus
    result = Image.composite(current_image, soft_focus, mask)

    # Draw border if requested
    if rect_border_width > 0:
        # Draw border on a transparent layer and composite over result
        border_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(border_layer)
        # Shrink rectangle by half the border width to keep border inside image
        shrink = rect_border_width // 2
        border_rect = (
            max(x1 + shrink, 0),
            max(y1 + shrink, 0),
            min(x2 - shrink, w_img - 1),
            min(y2 - shrink, h_img - 1)
        )
        # Ensure color is RGBA
        border_color_rgba = tuple(rect_border_color) + (255,)
        border_draw.rectangle(
            border_rect,
            outline=border_color_rgba,
            width=rect_border_width
        )
        # Convert result to RGBA if not already
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        result = Image.alpha_composite(result, border_layer).convert(current_image.mode)

    return result, {
        "blur_radius": blur_radius,
        "blend_alpha": blend_alpha,
        "rect": rect,
        "rect_abs": abs_rect,
        "rect_border_width": rect_border_width,
        "rect_border_color": rect_border_color
    }



def clouds1_fractal_noise(params, current_inputs):
    """
    Generates a classic fractal noise "clouds" image.

    The function creates a grayscale image using fractal noise, with adjustable scale and disorder.
    The output image size can be specified, defaulting to 1024x1024 pixels.

    Args:
        params (dict): Dictionary with optional keys:
            - 'scale' (float): Controls the size of the noise features. Range: 1.0 (large) to 8.0 (small). Default is 4.0.
            - 'disorder' (float): Controls the randomness/disorder of the noise. Range: 0.0 (smooth) to 1.0 (chaotic). Default is 0.5.
            - 'width' (int): Output image width in pixels. Default is 1024.
            - 'height' (int): Output image height in pixels. Default is 1024.
            - 'seed' (int): Optional random seed for reproducibility.
        current_inputs (list): Not used; included for interface compatibility.

    Returns:
        PIL.Image.Image: The generated fractal noise clouds image (mode 'L').
        dict: Parameters used, including resolved scale, disorder, width, height, and seed.
    """


    # Parameter defaults
    scale = float(params.get('scale', 4.0))
    scale = max(1.0, min(8.0, scale))
    disorder = float(params.get('disorder', 0.5))
    disorder = max(0.0, min(1.0, disorder))
    width = int(params.get('width', 1024))
    height = int(params.get('height', 1024))
    seed = params.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Fractal noise parameters
    octaves = int(np.clip(scale, 1, 8))
    persistence = 0.5 + 0.5 * disorder  # 0.5 (smooth) to 1.0 (chaotic)
    lacunarity = 2.0

    def perlin_noise(shape, res):
        """Simple Perlin-like noise using numpy interpolation."""
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.stack(np.meshgrid(np.arange(res[0]+1), np.arange(res[1]+1)), axis=-1)
        gradients = np.random.randn(res[0]+1, res[1]+1, 2)
        gradients /= np.linalg.norm(gradients, axis=2, keepdims=True)
        xs = np.linspace(0, res[0], shape[0], endpoint=False)
        ys = np.linspace(0, res[1], shape[1], endpoint=False)
        xi = xs.astype(int)
        yi = ys.astype(int)
        xf = xs - xi
        yf = ys - yi
        u = f(xf)
        v = f(yf)
        result = np.zeros(shape)
        for i in range(2):
            for j in range(2):
                g = gradients[xi + i, yi + j]
                dx = xf - i
                dy = yf - j
                dot = g[...,0]*dx + g[...,1]*dy
                weight = ((1-u) if i==0 else u)[:,None] * ((1-v) if j==0 else v)
                result += dot * weight
        return result

    # Generate fractal noise
    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for o in range(octaves):
        res = (int(width // (2**o)), int(height // (2**o)))
        res = (max(1, res[0]), max(1, res[1]))
        n = perlin_noise((height, width), res)
        noise += n * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to 0-255
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    img = Image.fromarray((noise * 255).astype(np.uint8), mode='L')

    return img, {
        'scale': scale,
        'disorder': disorder,
        'width': width,
        'height': height,
        'seed': seed
    }

def classic_perlin(params, current_inputs):
    """
    Generates a 2D classic Perlin noise image using numpy for efficient computation.

    Args:
        params (dict): Dictionary with optional keys:
            - 'scale' (float): Controls the frequency of the noise. Higher values = more detail. Default is 8.0.
            - 'disorder' (float): Controls the randomness of the gradients. Default is 1.0.
            - 'image_size' (int or tuple): Output image size as int (for square) or (width, height). Default is 1024.
            - 'seed' (int): Optional random seed for reproducibility.

        current_inputs (list): Not used, included for interface compatibility.

    Returns:
        PIL.Image.Image: Grayscale image of Perlin noise.
        dict: Parameters used, including actual image size and seed.
    """
    import numpy as np
    from PIL import Image

    print('Computing perlin'*10)

    # Parse parameters
    scale = float(params.get('scale', 8.0))
    disorder = float(params.get('disorder', 1.0))
    image_size = params.get('image_size', 1024)
    seed = params.get('seed', None)

    # Determine image dimensions
    if isinstance(image_size, int):
        w, h = image_size, image_size
    else:
        w, h = int(image_size[0]), int(image_size[1])

    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Helper functions for Perlin noise
    def lerp(a, b, t):
        return a + t * (b - a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        # 8 possible directions
        vectors = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]])
        g = vectors[h % 8]
        return g[...,0]*x + g[...,1]*y

    # Generate permutation table
    perm = rng.integers(0, 256, size=256, dtype=np.int32)
    perm = np.concatenate([perm, perm])

    # Generate grid coordinates
    linx = np.linspace(0, scale, w, endpoint=False)
    liny = np.linspace(0, scale, h, endpoint=False)
    x, y = np.meshgrid(linx, liny)

    # Integer part of coordinates
    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255

    # Fractional part
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    u = fade(xf)
    v = fade(yf)

    # Hash coordinates of the 4 corners
    aa = perm[perm[xi    ] + yi    ]
    ab = perm[perm[xi    ] + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]

    # Add disorder to gradients
    disorder_scale = max(0.0, float(disorder))
    def grad(h, x, y):
        return gradient(h, x, y) + disorder_scale * (rng.random(x.shape) - 0.5)

    # Calculate gradients at corners
    n00 = grad(aa, xf    , yf    )
    n01 = grad(ab, xf    , yf - 1)
    n10 = grad(ba, xf - 1, yf    )
    n11 = grad(bb, xf - 1, yf - 1)

    # Interpolate
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    noise = lerp(x1, x2, v)

    # Normalize to [0,255]
    noise_min = noise.min()
    noise_max = noise.max()
    norm_noise = (noise - noise_min) / (noise_max - noise_min + 1e-8)
    img_array = (norm_noise * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode='L')
    return img, {
        'scale': scale,
        'disorder': disorder,
        'image_size': (w, h),
        'seed': seed
    }





def resize_stage(stage, current_image):
    """
    Resizes the current image to the specified size.

    Args:
        stage (dict): Should contain one or more of:
            - 'size': (width, height) tuple
            - 'width' and/or 'height': target dimensions (aspect ratio preserved if only one is given)
            - 'width_ratio' and/or 'height_ratio': scale factors for width/height (e.g., 0.5 for half size)
            - 'resample': PIL resampling filter (default: Image.LANCZOS)
        current_image (PIL.Image.Image): The image to resize.

    Returns:
        PIL.Image.Image: The resized image.
    """
    resample = stage.get('resample', Image.LANCZOS)
    orig_width, orig_height = current_image.size

    # Direct size tuple
    if 'size' in stage and stage['size']:
        new_size = stage['size']
    else:
        # Ratios
        width_ratio = stage.get('width_ratio')
        height_ratio = stage.get('height_ratio')
        width = stage.get('width')
        height = stage.get('height')

        # Compute new width
        if width is not None:
            new_width = width
        elif width_ratio is not None:
            new_width = int(orig_width * width_ratio)
        else:
            new_width = None

        # Compute new height
        if height is not None:
            new_height = height
        elif height_ratio is not None:
            new_height = int(orig_height * height_ratio)
        else:
            new_height = None

        # Aspect ratio logic
        if new_width is not None and new_height is not None:
            new_size = (new_width, new_height)
        elif new_width is not None:
            scale = new_width / orig_width
            new_size = (new_width, int(orig_height * scale))
        elif new_height is not None:
            scale = new_height / orig_height
            new_size = (int(orig_width * scale), new_height)
        else:
            raise ValueError("resize_stage: Must specify 'size', 'width', 'height', 'width_ratio', or 'height_ratio'.")

    return current_image.resize(new_size, resample=resample)



def pass_through(stage, current_inputs): 
    momeutils.crline('Passing through, no processing was applied to the image')
    return current_inputs[0]['image'], {}


def crop_stage(stage, current_inputs):
    """
    Crops the current image to the specified box, supporting both absolute and relative coordinates.

    Args:
        stage (dict): May contain:
            - 'box': (left, upper, right, lower) tuple (absolute or relative floats)
            - 'left', 'upper', 'right', 'lower': coordinates (absolute or relative floats)
            - 'from_top', 'from_bottom', 'from_left', 'from_right': crop distances (absolute or relative floats)
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parents data

    Returns:
        PIL.Image.Image: The cropped image.
        dict: Metadata about the crop box used.
    """
    current_image = current_inputs[0]['image']
    width, height = current_image.width, current_image.height

    def resolve(val, maxval):
        """Convert relative (float) or absolute (int) coordinate to absolute pixel value."""
        if isinstance(val, float):
            return int(round(val * maxval))
        return int(val)

    # Priority: from_* controls > box > individual sides
    if any(k in stage for k in ['from_top', 'from_bottom', 'from_left', 'from_right']):
        from_top = resolve(stage.get('from_top', 0), height)
        from_bottom = resolve(stage.get('from_bottom', 0), height)
        from_left = resolve(stage.get('from_left', 0), width)
        from_right = resolve(stage.get('from_right', 0), width)
        left = from_left
        upper = from_top
        right = width - from_right
        lower = height - from_bottom
        box = (left, upper, right, lower)
    elif 'box' in stage:
        # box can be absolute or relative
        box = stage['box']
        box = tuple(
            resolve(val, width if i % 2 == 0 else height)
            for i, val in enumerate(box)
        )
    else:
        left = resolve(stage.get('left', 0), width)
        upper = resolve(stage.get('upper', 0), height)
        right = resolve(stage.get('right', width), width)
        lower = resolve(stage.get('lower', height), height)
        box = (left, upper, right, lower)

    # Ensure box is within image bounds
    box = (
        max(0, min(box[0], width)),
        max(0, min(box[1], height)),
        max(0, min(box[2], width)),
        max(0, min(box[3], height)),
    )

    cropped = current_image.crop(box)
    return cropped, {"box": box}


def get_gemini_image(stage, current_inputs): 


    # make tmp_images
    im_paths = []
    for i, ci in enumerate(current_inputs): 
        current_im = ci['image']

        p = os.path.join("/tmp", "im_{}.png".format(i))
        # Resize image if larger than 1024px in either dimension
        max_dim = 1024
        w, h = current_im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            current_im = current_im.resize(new_size, Image.LANCZOS)
        current_im.save(p)
        im_paths.append(p)
    
    prompt = stage['prompt']   
    generated_path = momeutils.make_gemini_image(prompt, im_paths)
    for im_path in im_paths: 
        os.remove(im_path)
    # generated_path = "/home/mehdimounsif/Images/generated/last_image.png"
    return Image.open(generated_path)   , {"prompt": prompt}



def overlay_stage(stage, current_inputs):
    """
    Overlays one parent image onto another.

    Args:
        stage (dict): Should contain:
            - 'to_overlay_parent_id': node id of the image to overlay (required)
            - 'position': (x, y) tuple for top-left corner, or string: 'center', 'bottom-right', etc. (default: (0, 0))
            - 'relative_position': (x_frac, y_frac) tuple, each in [0, 1], for relative placement
            - 'alpha': float in [0, 1] for overlay opacity (optional)
            - 'resize': (width, height) tuple to resize overlay (optional)
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parent images.

    Returns:
        (PIL.Image.Image, dict): The image with overlay applied, and metadata.
    """
    # Get base image (first parent by convention)
    # base_image = current_inputs[0]['image']

    # Find overlay image by parent id

    overlay_id = stage['to_overlay_parent_id']
    base_image = None 
    overlay_image = None
    for parent in current_inputs:
        if parent['id'] == overlay_id:
            overlay_image = parent['image']
            print('Overlay image: ', parent['id'])
        else: 
            print('Base image: ', parent['id'])
            base_image = parent['image']

        if base_image and overlay_image: 
            break 
    if overlay_image is None:
        raise ValueError(f"Overlay parent id '{overlay_id}' not found in current_inputs.")

    # Optionally resize the overlay
    resize = stage.get('resize', None)
    if resize:
        overlay_image = overlay_image.resize(resize, Image.LANCZOS)

    # Optionally adjust overlay alpha
    alpha = stage.get('alpha', None)
    if alpha is not None:
        overlay_image = overlay_image.copy()
        alpha_mask = overlay_image.split()[-1].point(lambda p: int(p * alpha))
        overlay_image.putalpha(alpha_mask)

    base_w, base_h = base_image.size
    ovl_w, ovl_h = overlay_image.size

    # Determine position
    position = stage.get('position', None)
    relative_position = stage.get('relative_position', None)
    if position is not None:
        if isinstance(position, str):
            if position == "center":
                position = ((base_w - ovl_w) // 2, (base_h - ovl_h) // 2)
            elif position == "bottom-right":
                position = (base_w - ovl_w, base_h - ovl_h)
            elif position == "top-right":
                position = (base_w - ovl_w, 0)
            elif position == "bottom-left":
                position = (0, base_h - ovl_h)
            elif position == "top-left":
                position = (0, 0)
            else:
                position = (0, 0)
        # else: assume tuple of ints
    elif relative_position is not None:
        x_frac, y_frac = relative_position
        x = int(x_frac * (base_w - ovl_w))
        y = int(y_frac * (base_h - ovl_h))
        position = (x, y)
    else:
        position = (0, 0)

    # Ensure base image is RGBA
    base = base_image.convert("RGBA")
    composite = base.copy()
    composite.paste(overlay_image, position, mask=overlay_image)

    # Prepare metadata
    meta = {
        "to_overlay_parent_id": overlay_id,
        "position": position,
        "relative_position": relative_position,
        "alpha": alpha,
        "resize": resize
    }

    return composite, meta

def warp_stage(stage, current_inputs):
    """
    Warps an input image by displacing its pixels according to the slopes of a gradient image,
    scaled by an intensity factor.

    Args:
        stage (dict): Should contain:
            - 'input': node id of the image to be warped (required)
            - 'gradient': node id of the gradient image (required)
            - 'intensity': float, scaling factor for the displacement (default: 1.0)
            - 'resize_gradient': bool, whether to resize gradient to input image size (default: True)
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parent images.

    Returns:
        (PIL.Image.Image, dict): The warped image, and metadata.
    """
    from PIL import Image
    import numpy as np

    input_id = stage['input']
    gradient_id = stage['gradient']
    intensity = stage.get('intensity', 1.0)
    resize_gradient = stage.get('resize_gradient', True)

    input_image = None
    gradient_image = None

    for parent in current_inputs:
        if parent['id'] == input_id:
            input_image = parent['image']
        elif parent['id'] == gradient_id:
            gradient_image = parent['image']
        if input_image and gradient_image:
            break

    if input_image is None:
        raise ValueError(f"Input image id '{input_id}' not found in current_inputs.")
    if gradient_image is None:
        raise ValueError(f"Gradient image id '{gradient_id}' not found in current_inputs.")

    input_image = input_image.convert('RGBA')
    if resize_gradient:
        gradient_image = gradient_image.resize(input_image.size, Image.BILINEAR)
    else:
        if gradient_image.size != input_image.size:
            raise ValueError("Gradient image size must match input image size or set 'resize_gradient' to True.")

    # Convert images to numpy arrays
    input_np = np.array(input_image)
    grad_np = np.array(gradient_image.convert('L'))  # Use luminance for gradient

    # Compute gradients (slopes) using Sobel operator
    from scipy.ndimage import sobel

    grad_x = sobel(grad_np, axis=1, mode='reflect')
    grad_y = sobel(grad_np, axis=0, mode='reflect')

    # Normalize gradients to [-1, 1]
    grad_x = grad_x / (np.max(np.abs(grad_x)) + 1e-8)
    grad_y = grad_y / (np.max(np.abs(grad_y)) + 1e-8)

    # Scale by intensity
    dx = grad_x * intensity
    dy = grad_y * intensity

    # Generate meshgrid of coordinates
    h, w = grad_np.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).clip(0, w - 1)
    map_y = (y + dy).clip(0, h - 1)

    # Remap pixels using bilinear interpolation
    def bilinear_remap(img, map_x, map_y):
        from scipy.ndimage import map_coordinates
        remapped = np.zeros_like(img)
        for c in range(img.shape[2]):
            remapped[..., c] = map_coordinates(img[..., c], [map_y, map_x], order=1, mode='reflect')
        return remapped

    warped_np = bilinear_remap(input_np, map_x, map_y)
    warped_img = Image.fromarray(np.uint8(warped_np))

    meta = {
        'input': input_id,
        'gradient': gradient_id,
        'intensity': intensity,
        'resize_gradient': resize_gradient
    }
    return (warped_img, meta)


# def compose(stages, base_image=None):
#     """
#     Compose an image from a list of stage dicts.
#     Each stage dict must have a 'type' key: 'image', 'text', 'bars', etc.
#     """
#     current_image = None
#     last_text_info = None

#     for stage in stages:
#         stage_type = stage['type']
#         if stage_type == 'image':
#             current_image = load_image_stage(stage, current_image)
#         elif stage_type == 'text':
#             current_image, last_text_info = add_text(stage, current_image)
#         elif stage_type == 'bars':
#             if last_text_info is None:
#                 raise ValueError("No previous text info for bars placement.")
#             current_image = bars_stage(stage, current_image, last_text_info)
#         elif stage_type == "vhs_scanline": 
#             current_image = vhs_scanline(stage, current_image)
#         elif stage_type == "blur": 
#             current_image= soft_focus_blur(stage, current_image)
#         elif stage_type == "vhs_grain": 
#             current_image = vhs_grain(stage, current_image)
#         elif stage_type == "resize": 
#             current_image = resize_stage(stage, current_image)
#         elif stage_type == "crop": 
#             current_image = crop_stage(stage, current_image)
#         elif stage_type == "overlay": 
#             current_image = overlay_stage(stage, current_image)
#         elif stage_type == "raw": 
#             current_image = stage['pil_image']
#         else:
#             raise ValueError(f"Unknown stage type: {stage_type}")

#     return current_image


# def compose(source_folder, G, outputs=None, base_image=None, cache=None, invalidate=True):
#     """
#     Incremental DAG execution with caching and content-hash validation.

#     Cache structure:
#         cache[node_id] = (image, meta, content_hash)
#     """

#     if not isinstance(G, nx.DiGraph):
#         raise TypeError("compose expects a networkx.DiGraph")

#     if cache is None:
#         cache = {}

#     # Resolve targets
#     if outputs is None:
#         target_nodes = {n for n in G.nodes if G.out_degree(n) == 0}
#         if not target_nodes:
#             raise ValueError("No sink nodes found; specify outputs explicitly or ensure the graph is a DAG with sinks.")
#     else:
#         target_nodes = set(outputs)
#         missing = [n for n in target_nodes if n not in G]
#         if missing:
#             raise KeyError(f"Requested output nodes not present in the graph: {missing}")

#     # Induced subgraph (only what we need for targets)
#     required_nodes: set = set()
#     for out in target_nodes:
#         required_nodes.add(out)
#         required_nodes.update(nx.ancestors(G, out))
#     H = G.subgraph(required_nodes).copy()

#     # --- Invalidate nodes outside required subgraph ---
#     if invalidate:
#         for nid in list(cache.keys()):
#             if nid not in H.nodes:
#                 print('Removing from cache,', nid)
#                 cache.pop(nid, None)

#     # Topological order
#     topo = list(nx.topological_sort(H))

#     # --- Core execution ---
#     for n in topo:
#         node_attrs = H.nodes[n]
#         if "node_content" not in node_attrs:
#             raise KeyError(f"Node {n!r} is missing 'node_content' attribute.")

#         node_content = node_attrs["node_content"]


#         print("node_content before hash: {}".format(node_content.strip()))
#         # cleaning node_content for hash by only selecting structured data 
#         try: 
#             node_content_for_hash = momeutils.parse_json(node_content)
#         except Exception as e: 
#             node_content_for_hash = node_content
#         print("node_content for hash: {}".format(node_content_for_hash))

#         content_hash = _hash_content(node_content_for_hash)

#         # Reuse only if cached content matches
#         if n in cache and cache[n][2] == content_hash and n not in outputs:
#             print('Skipping node ', n)
#             if n in outputs: 
#                 print('I should not have skipped')
#             continue
#         else: 
#             if n in cache: 
#                 print(node_content.strip())
#                 print('Content hash problem', n, content_hash, cache[n][2])
#                 # input(' ok ? ')
#             momeutils.crline(f'Computation to run for node {n}')

#         # Build inputs
#         parent_ids = sorted(H.predecessors(n), key=lambda x: str(x))
#         inputs = [
#             {"id": pid, "image": cache[pid][0], "meta": cache[pid][1]}
#             for pid in parent_ids
#         ]

#         # Execute node
#         image, meta = execute_node(source_folder, n, node_content, inputs, base_image)
#         if meta is None:
#             meta = {}
#         elif not isinstance(meta, dict):
#             raise TypeError(f"execute_node must return a dict for meta; got {type(meta)} at node {n!r}.")

#         cache[n] = (image, meta, content_hash)

#     # Collect outputs
#     final_nodes = list(target_nodes) if outputs is not None else [n for n in H.nodes if H.out_degree(n) == 0]
#     result = {nid: cache[nid][0] for nid in final_nodes}


#     return result, cache


# If a parent node is modified, the results must also be returned, even if it wasn't originally part of 'outputs'

def compose(source_folder, G, outputs=None, base_image=None, cache=None, invalidate=True):
    """
    Incremental DAG execution with caching and content-hash validation.

    Cache structure:
        cache[node_id] = (image, meta, content_hash)

    Returns:
        result: dict mapping node_id to image for all requested outputs and any recomputed parent nodes.
        cache: updated cache.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("compose expects a networkx.DiGraph")

    if cache is None:
        cache = {}

    # Resolve targets
    if outputs is None:
        target_nodes = {n for n in G.nodes if G.out_degree(n) == 0}
        if not target_nodes:
            raise ValueError("No sink nodes found; specify outputs explicitly or ensure the graph is a DAG with sinks.")
    else:
        target_nodes = set(outputs)
        missing = [n for n in target_nodes if n not in G]
        if missing:
            raise KeyError(f"Requested output nodes not present in the graph: {missing}")

    # Induced subgraph (only what we need for targets)
    required_nodes: set = set()
    for out in target_nodes:
        required_nodes.add(out)
        required_nodes.update(nx.ancestors(G, out))
    H = G.subgraph(required_nodes).copy()

    # --- Invalidate nodes outside required subgraph ---
    if invalidate:
        for nid in list(cache.keys()):
            if nid not in H.nodes:
                print('Removing from cache,', nid)
                cache.pop(nid, None)

    # Topological order
    topo = list(nx.topological_sort(H))

    # Track recomputed nodes
    recomputed_nodes = set()

    # --- Core execution ---
    for n in topo:
        node_attrs = H.nodes[n]
        if "node_content" not in node_attrs:
            raise KeyError(f"Node {n!r} is missing 'node_content' attribute.")

        node_content = node_attrs["node_content"]

        print("node_content before hash: {}".format(node_content.strip()))
        # cleaning node_content for hash by only selecting structured data 
        try: 
            node_content_for_hash = momeutils.parse_json(node_content)
        except Exception as e: 
            node_content_for_hash = node_content
        print("node_content for hash: {}".format(node_content_for_hash))

        content_hash = _hash_content(node_content_for_hash)

        # Reuse only if cached content matches
        if n in cache and cache[n][2] == content_hash and n not in outputs:
            print('Skipping node ', n)
            if n in outputs: 
                print('I should not have skipped')
            continue
        else: 
            if n in cache: 
                print(node_content.strip())
                print('Content hash problem', n, content_hash, cache[n][2])
                # input(' ok ? ')
            momeutils.crline(f'Computation to run for node {n}')
            recomputed_nodes.add(n)  # Mark as recomputed

        # Build inputs
        parent_ids = sorted(H.predecessors(n), key=lambda x: str(x))
        inputs = [
            {"id": pid, "image": cache[pid][0], "meta": cache[pid][1]}
            for pid in parent_ids
        ]

        # Execute node
        image, meta = execute_node(source_folder, n, node_content, inputs, base_image)
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f"execute_node must return a dict for meta; got {type(meta)} at node {n!r}.")

        cache[n] = (image, meta, content_hash)

    # Collect outputs: all requested outputs + all recomputed nodes
    final_nodes = set(target_nodes) | recomputed_nodes
    result = {nid: cache[nid][0] for nid in final_nodes}

    return result, cache


def blend_overlay_stage(stage, current_inputs):
    """
    Blends two images using a grayscale map as the blend factor.

    Args:
        stage (dict): Should contain:
            - 'top': node id of the top image (required)
            - 'bottom': node id of the bottom image (required)
            - 'map': node id of the grayscale blend map (required)
        current_inputs (list): List of dicts (with keys 'id', 'image', 'meta') containing parent images.

    Returns:
        (PIL.Image.Image, dict): The blended image and metadata.
    """
    top_id = stage['top']
    bottom_id = stage['bottom']
    map_id = stage['map']

    top_image = None
    bottom_image = None
    map_image = None

    for parent in current_inputs:
        if parent['id'] == top_id:
            top_image = parent['image']
        elif parent['id'] == bottom_id:
            bottom_image = parent['image']
        elif parent['id'] == map_id:
            map_image = parent['image']
        if top_image and bottom_image and map_image:
            break

    if top_image is None:
        raise ValueError(f"Top image id '{top_id}' not found in current_inputs.")
    if bottom_image is None:
        raise ValueError(f"Bottom image id '{bottom_id}' not found in current_inputs.")
    if map_image is None:
        raise ValueError(f"Map image id '{map_id}' not found in current_inputs.")

    # Ensure all images are the same size
    size = bottom_image.size
    if top_image.size != size:
        top_image = top_image.resize(size, Image.LANCZOS)
    if map_image.size != size:
        map_image = map_image.resize(size, Image.LANCZOS)

    # Convert images to RGBA
    top_rgba = top_image.convert('RGBA')
    bottom_rgba = bottom_image.convert('RGBA')
    # Convert map to single-channel grayscale
    blend_map = map_image.convert('L')

    # Prepare alpha mask from blend map
    # The blend map acts as the alpha for the top image
    # 0 = only bottom, 255 = only top
    blended = Image.composite(top_rgba, bottom_rgba, blend_map)

    meta = {
        'top': top_id,
        'bottom': bottom_id,
        'map': map_id
    }
    return (blended, meta)


def execute_node(source_folder, node_id, node_contents, current_inputs, base_image = None): 
    
 
    data = {}
    data = momeutils.outer_parse_generic(node_contents) 
    available_tools = json.load(open(os.path.join(os.path.dirname(__file__), "mOmEdRiAn_available_tools.json")))

    if data is None: 
        momeutils.crline('Fallback for node id {}'.format(node_id, node_contents))
        match = re.search(r'!\[\[(.*?)\]\]', node_contents)
        if match:
            img_path = match.group(1)
        img_path = os.path.join(source_folder, img_path) 
        img = Image.open(img_path).convert('RGBA')
        return img, {}
    
    else: 
        processing_func = available_tools.get(data.get('type', 'fallback'), 'fallback')['function']
        momeutils.crline(f'Processing func for {node_id}: {processing_func}')
        func = getattr(sys.modules[__name__], processing_func)
        img, meta = func(data, current_inputs)
    
    return img, meta



def execute_composition(composition_path):

    composition_data = json.load(open(composition_path)) 
    img = compose(composition_data)
    return img

def produce_composition(data_path, selected_ids): 
    
    """
    Generates and saves image compositions for selected nodes in a graph.
    This function performs the following steps:
    1. Loads graph data from a JSON file at `data_path`.
    2. Constructs a graph object from the loaded data.
    3. Determines the source folder for intermediate images.
    4. Composes images for nodes specified by `selected_ids`.
    5. Saves the composed images to disk and updates the corresponding node's text field
       to reference the saved image.
    6. Ensures all image resources are properly closed to free memory.
    7. Writes the updated graph data back to the original JSON file.
    Args:
        data_path (str): Path to the JSON file containing graph data.
        selected_ids (list): List of node IDs for which compositions should be generated.
    Returns:
        None
    """
    
    data = json.load(open(data_path))
    graph =  hub_utils.make_graph_from_canva(data)
    img_source_folder = os.path.join(os.path.dirname(data_path), os.path.basename(data_path).split('.')[0] + "_interm")
    cached = {}
    cache_path = os.path.join(os.path.dirname(__file__), "current_cache.pkl")
    if os.path.exists(cache_path): 
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    

    results, cached = compose(img_source_folder, graph, selected_ids, cache= cached)
    print(results)
    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)

    for k in results.keys(): 
        print(f'Results for node {k}')
        final_path = os.path.join(img_source_folder, f"interm_{k}.png")
        results[k].save(final_path)
        for n in data['nodes']: 
            if n['id'] == k: 
                n['text'] = re.sub(r'!\[\[.*?\]\]', '', n['text']).strip()
                n['text'] += f"\n\n![[{os.path.basename(final_path)}]]"
        _safe_close_image(results[k])
    
    results.clear()
    del results
    with open(data_path, "w") as f: 
        json.dump(data, f, indent = 4)

def _safe_close_image(img) -> None:
    """
    Attempt to close a PIL Image (or compatible object) without raising.
    Closing helps release file handles and memory for large images sooner.
    """
    try:
        close = getattr(img, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


# Update this function: main idea is that I want to go through every the hierarchy of evey selected node to prepare all computations
# It is a directed graph but several paths may be computed to reach final node

if __name__ == "__main__":
    # Example usage: three texts, bars on first and third, with various positioning
    im_path = os.path.join(os.path.expanduser('~'), "Images", "thumbnails_sources", "IMG_2185.JPG")
    stages = [
        {"type": "image", "path": im_path},
        {"type": "resize", "width_ratio": 0.5},
        {"type": "blur", "blur_radius": 45.},
        {"type": "vhs_grain"}, 
        {"type": "vhs_scanline", "line_opacity" : 150}, 
        # {"type": "text", "text": "LaToile: Psychohistory and System Dynamics", "font_path" : "/home/mehdimounsif/Documents/Fonts/Ruina.ttf", "font_size": 120, "bars": True, "y": 0.65, "x": 0.5, "color": (255, 233, 92, 255)},  
        {"type": "text", "text": "LaToile", "font_path" : "/home/mehdimounsif/Documents/Fonts/Ruina.ttf", "font_size": 240, "bars": False, "y": 0.3, "x": 0.8, "color": (255, 233, 255, 255)},  
        {"type": "text", "text": "Psychohistory and System Dynamics", "font_path" : "/home/mehdimounsif/Documents/Fonts/Ruina.ttf", "font_size": 150, "bars": True, "y": 0.65, "x": 0.5, "color": (255, 233, 255, 255)},  
    ]

    comp_path = os.path.join(os.path.dirname(__file__), "tmp_comp.json")
    with open(comp_path) as f: 
        json.dump(stages, f, identnt = 4)
    img = execute_composition(comp_path)
    # img = compose(stages)

  
    # overlay_stages = [
    # {"type": "raw", "pil_image": img},
    # {
    #     "type": "overlay",
    #     "overlay_pil": img,
    #     "resize": (350, 200),
    #     "relative_position": (0.5,0.65),
        
    # },
    # ]


    # composed = compose(overlay_stages)
    # composed.show()    

    # img.show()
    img= img.convert("RGB")
    img.save(os.path.join(os.path.dirname(im_path), "mome_composition_results.jpg"))


