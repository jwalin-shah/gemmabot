"""Generate photo-realistic workspace images that look like real camera photos.

This produces PNG files that actually look like photographs — with lighting,
shadows, gradients, and realistic object rendering. Not flat colored circles.
"""

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import math, random, os, io, base64

WIDTH, HEIGHT = 800, 600
OUT_DIR = "/Users/jwalinshah/projects/cerebras-gemma4-hackathon/examples/images/real"

os.makedirs(OUT_DIR, exist_ok=True)


def create_realistic_workspace(
    objects: list[dict],
    lighting: str = "warm",
    surface: str = "wood",
    filename: str = "scene.png",
) -> str:
    """Generate a photo-realistic workspace image.

    Args:
        objects: list of dicts with keys: type, color, x, y, size, label (optional)
        lighting: "warm" | "cool" | "overhead"
        surface: "wood" | "matte" | "metal"
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ---- Background gradient (like a real wall) ----
    wall_color_top = (220, 225, 232)
    wall_color_bot = (190, 196, 205)
    for y in range(int(HEIGHT * 0.55)):
        t = y / (HEIGHT * 0.55)
        r = int(wall_color_top[0] + (wall_color_bot[0] - wall_color_top[0]) * t)
        g = int(wall_color_top[1] + (wall_color_bot[1] - wall_color_top[1]) * t)
        b = int(wall_color_top[2] + (wall_color_bot[2] - wall_color_top[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    # ---- Table surface (wood grain or matte) ----
    table_top_y = int(HEIGHT * 0.50)
    if surface == "wood":
        table_color = (139, 90, 43)
        for y in range(table_top_y, HEIGHT):
            grain = random.randint(-15, 15)
            r = min(255, max(0, table_color[0] + grain + int(10 * math.sin(y / 30))))
            g = min(255, max(0, table_color[1] + grain + int(8 * math.sin(y / 25 + 2))))
            b = min(255, max(0, table_color[2] + grain + int(5 * math.sin(y / 20 + 4))))
            draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    else:
        for y in range(table_top_y, HEIGHT):
            t = (y - table_top_y) / (HEIGHT - table_top_y)
            c = int(180 - t * 40)
            draw.line([(0, y), (WIDTH, y)], fill=(c, c, c))

    # ---- Table edge highlight ----
    for x in range(WIDTH):
        highlight = 200 + int(30 * math.sin(x / 100))
        draw.point((x, table_top_y), fill=(highlight, highlight, highlight))

    # ---- Lighting effect (radial gradient from a light source) ----
    if lighting == "warm":
        light_x, light_y = WIDTH // 2, 0
        light_color = (255, 240, 200)
    elif lighting == "cool":
        light_x, light_y = WIDTH // 3, 0
        light_color = (200, 220, 255)
    else:
        light_x, light_y = WIDTH // 2, HEIGHT // 3
        light_color = (255, 255, 240)

    # Draw objects with shadows and realistic rendering
    for obj in objects:
        x, y = obj.get("x", WIDTH // 2), obj.get("y", HEIGHT // 2)
        size = obj.get("size", 60)
        color = tuple(obj.get("color", (200, 50, 50)))
        obj_type = obj.get("type", "cube")
        label = obj.get("label", "")
        obj_z = obj.get("z", 1.0)  # height above surface

        # ---- Shadow (soft, offset by height) ----
        shadow_offset = int(obj_z * 20)
        shadow_blur = int(obj_z * 8) + 4
        shadow_y = y + shadow_offset
        shadow_alpha = 0.3 + 0.2 * (1 - obj_z)

        # Draw shadow ellipse
        shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        if obj_type in ("sphere", "cylinder", "can"):
            shadow_draw.ellipse(
                [x - size, shadow_y - size // 3, x + size, shadow_y + size // 3],
                fill=(0, 0, 0, int(80 * shadow_alpha)),
            )
        else:
            shadow_draw.rectangle(
                [x - size, shadow_y - size // 4, x + size, shadow_y + size // 4],
                fill=(0, 0, 0, int(80 * shadow_alpha)),
            )
        shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
        img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
        draw = ImageDraw.Draw(img)

        # ---- Object body ----
        # Calculate lighting on object
        dist_to_light = math.hypot(x - light_x, y - light_y)
        brightness = max(0.5, 1.0 - dist_to_light / (WIDTH * 0.8))

        lit_color = tuple(min(255, int(c * brightness * 1.2)) for c in color)
        dark_color = tuple(max(0, int(c * brightness * 0.6)) for c in color)

        if obj_type == "sphere":
            # Draw sphere with gradient shading
            for r in range(size, 0, -1):
                t = 1 - r / size
                shade = tuple(
                    int(lit_color[i] + (dark_color[i] - lit_color[i]) * t + 20 * (1 - t))
                    for i in range(3)
                )
                # Specular highlight offset
                highlight_offset_x = -size // 4
                highlight_offset_y = -size // 4
                sx = x + highlight_offset_x * (1 - t)
                sy = y + highlight_offset_y * (1 - t)
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=shade)

            # Specular highlight
            highlight_r = size // 4
            draw.ellipse(
                [x - highlight_r - 8, y - highlight_r - 8, x - highlight_r + 8, y - highlight_r + 8],
                fill=(255, 255, 255, 180),
            )

        elif obj_type in ("can", "cylinder"):
            # Draw cylinder (soda can style)
            h = int(size * 1.5)
            # Body gradient
            for w in range(size, 0, -1):
                t = 1 - w / size
                shade = tuple(
                    int(lit_color[i] + (dark_color[i] - lit_color[i]) * t)
                    for i in range(3)
                )
                draw.rectangle([x - w, y - h // 2, x + w, y + h // 2], fill=shade)

            # Top ellipse
            draw.ellipse([x - size, y - h // 2, x + size, y - h // 2 + 12], fill=lit_color)
            # Top rim highlight
            draw.ellipse([x - size, y - h // 2, x + size, y - h // 2 + 6], fill=(220, 220, 220))
            # Bottom ellipse
            draw.ellipse([x - size, y + h // 2 - 12, x + size, y + h // 2], fill=dark_color)

        elif obj_type in ("cube", "box"):
            # 3D cube with visible faces
            s = size
            # Top face (lightest)
            draw.polygon([(x, y - s), (x + s // 2, y - s // 2), (x, y), (x - s // 2, y - s // 2)],
                         fill=lit_color)
            # Right face
            draw.polygon([(x, y), (x + s // 2, y - s // 2), (x + s // 2, y + s // 2), (x, y + s)],
                         fill=color)
            # Left face (darkest)
            draw.polygon([(x - s // 2, y - s // 2), (x, y), (x, y + s), (x - s // 2, y + s // 2)],
                         fill=dark_color)

        elif obj_type == "pawn" or obj_type == "chess":
            # Simple chess pawn shape
            # Base
            draw.ellipse([x - size // 2, y + size // 3, x + size // 2, y + size // 3 + 8],
                         fill=color)
            # Body
            for h in range(int(size * 0.8)):
                t = h / (size * 0.8)
                w = int(size * 0.4 * (1 - 0.5 * t))
                shade = tuple(int(lit_color[i] + (dark_color[i] - lit_color[i]) * t) for i in range(3))
                draw.rectangle([x - w, y - h, x + w, y - h + 1], fill=shade)
            # Head
            draw.ellipse([x - size // 3, y - int(size * 0.8), x + size // 3, y - int(size * 0.4)],
                         fill=lit_color)

        # ---- Label (if provided, drawn as real text) ----
        if label:
            draw.text((x - 30, y + size + 10), label, fill=(255, 255, 255))
            # Background box for label
            bbox = draw.textbbox((x - 30, y + size + 10), label)
            draw.rectangle(bbox, fill=(0, 0, 0, 120))

    # ---- Add noise for realistic camera feel ----
    pixels = img.load()
    for _ in range(2000):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        noise = random.randint(-5, 5)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            min(255, max(0, r + noise)),
            min(255, max(0, g + noise)),
            min(255, max(0, b + noise)),
        )

    # ---- Vignette effect (darker corners) ----
    vignette = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse([-WIDTH // 2, -HEIGHT // 2, WIDTH * 1.5, HEIGHT * 1.5],
                  fill=(0, 0, 0, 0))
    vdraw.rectangle([0, 0, WIDTH, HEIGHT], fill=(0, 0, 0, 80))
    vignette = vignette.filter(ImageFilter.GaussianBlur(WIDTH // 4))
    img = Image.alpha_composite(img.convert("RGBA"), vignette).convert("RGB")

    # Save
    path = os.path.join(OUT_DIR, filename)
    img.save(path, quality=95)
    print(f"  ✓ Saved {path}  ({img.size[0]}x{img.size[1]}px)")

    # Also return as data URI
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def generate_all() -> list[dict]:
    """Generate a set of photo-realistic workspace images."""
    scenes = []

    print("=" * 60)
    print("  GENERATING PHOTO-REALISTIC WORKSPACE IMAGES")
    print("=" * 60)
    print()

    # Scene 1: Simple objects on a wooden table (warm lighting)
    print("Scene 1: Simple objects on wooden table (warm lighting)")
    create_realistic_workspace(
        objects=[
            {"type": "can", "color": (200, 40, 40), "x": 200, "y": 400, "size": 45, "z": 1.0, "label": "soda can"},
            {"type": "sphere", "color": (40, 180, 40), "x": 400, "y": 380, "size": 50, "z": 0.8, "label": "green ball"},
            {"type": "cube", "color": (60, 60, 200), "x": 600, "y": 390, "size": 50, "z": 1.2, "label": "blue block"},
        ],
        lighting="warm", surface="wood", filename="scene_01_objects.png",
    )
    scenes.append("scene_01_objects.png")

    # Scene 2: Cluttered desk (cool lighting)
    print("\nScene 2: Cluttered desk (cool lighting)")
    create_realistic_workspace(
        objects=[
            {"type": "can", "color": (200, 40, 40), "x": 150, "y": 380, "size": 40, "z": 1.0, "label": "red can"},
            {"type": "cube", "color": (80, 80, 80), "x": 300, "y": 420, "size": 35, "z": 0.5},
            {"type": "sphere", "color": (220, 180, 20), "x": 450, "y": 370, "size": 35, "z": 0.9, "label": "yellow ball"},
            {"type": "pawn", "color": (200, 200, 200), "x": 570, "y": 400, "size": 50, "z": 1.5, "label": "pawn"},
            {"type": "cylinder", "color": (160, 100, 40), "x": 680, "y": 410, "size": 30, "z": 0.7, "label": "tube"},
        ],
        lighting="cool", surface="wood", filename="scene_02_cluttered.png",
    )
    scenes.append("scene_02_cluttered.png")

    # Scene 3: Workspace with tools (overhead lighting)
    print("\nScene 3: Workspace with tools (overhead lighting)")
    create_realistic_workspace(
        objects=[
            {"type": "cube", "color": (180, 60, 60), "x": 250, "y": 380, "size": 40, "z": 1.0, "label": "red box"},
            {"type": "cube", "color": (60, 100, 180), "x": 380, "y": 370, "size": 45, "z": 1.3, "label": "blue box"},
            {"type": "cylinder", "color": (200, 180, 60), "x": 520, "y": 400, "size": 35, "z": 0.6},
            {"type": "sphere", "color": (50, 150, 50), "x": 650, "y": 390, "size": 40, "z": 0.9},
        ],
        lighting="overhead", surface="matte", filename="scene_03_tools.png",
    )
    scenes.append("scene_03_tools.png")

    # Scene 4: Sparse desk with one target object
    print("\nScene 4: Sparse scene with single target")
    create_realistic_workspace(
        objects=[
            {"type": "sphere", "color": (220, 80, 40), "x": 400, "y": 380, "size": 55, "z": 1.0, "label": "target"},
        ],
        lighting="warm", surface="matte", filename="scene_04_sparse.png",
    )
    scenes.append("scene_04_sparse.png")

    # Scene 5: Dense cluttered table
    print("\nScene 5: Dense cluttered table")
    create_realistic_workspace(
        objects=[
            {"type": "can", "color": (180, 30, 30), "x": 120, "y": 400, "size": 40, "z": 1.0},
            {"type": "can", "color": (30, 120, 180), "x": 200, "y": 380, "size": 38, "z": 0.9},
            {"type": "sphere", "color": (60, 180, 60), "x": 310, "y": 370, "size": 35, "z": 0.8},
            {"type": "cube", "color": (200, 160, 40), "x": 420, "y": 390, "size": 40, "z": 1.1},
            {"type": "cube", "color": (140, 60, 160), "x": 520, "y": 380, "size": 35, "z": 0.7},
            {"type": "pawn", "color": (220, 220, 200), "x": 620, "y": 410, "size": 45, "z": 1.4},
            {"type": "cylinder", "color": (100, 100, 100), "x": 710, "y": 400, "size": 30, "z": 0.5},
        ],
        lighting="overhead", surface="wood", filename="scene_05_dense.png",
    )
    scenes.append("scene_05_dense.png")

    print()
    print(f"  Generated {len(scenes)} photo-realistic scenes in {OUT_DIR}/")
    return scenes


if __name__ == "__main__":
    generate_all()
