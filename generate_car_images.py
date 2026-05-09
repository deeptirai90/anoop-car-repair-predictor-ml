"""
generate_car_images.py
Creates 10 synthetic car incident images with damage visualizations
representing different severity levels and speeds.
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
import os

OUTPUT_DIR = "data/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INCIDENTS = [
    {"id": 1,  "speed": 15,  "damage": "Minor",    "cost": 12000,  "color": (220,50,50),   "parts": ["Front Bumper"]},
    {"id": 2,  "speed": 25,  "damage": "Minor",    "cost": 28000,  "color": (200,80,30),   "parts": ["Hood", "Headlight"]},
    {"id": 3,  "speed": 35,  "damage": "Moderate", "cost": 55000,  "color": (180,100,20),  "parts": ["Fender", "Door", "Bumper"]},
    {"id": 4,  "speed": 45,  "damage": "Moderate", "cost": 85000,  "color": (160,120,10),  "parts": ["Hood", "Radiator", "Airbag"]},
    {"id": 5,  "speed": 55,  "damage": "Severe",   "cost": 130000, "color": (140,140,0),   "parts": ["Engine Bay", "Frame", "Airbags"]},
    {"id": 6,  "speed": 65,  "damage": "Severe",   "cost": 185000, "color": (120,150,10),  "parts": ["Full Front", "Chassis", "Transmission"]},
    {"id": 7,  "speed": 75,  "damage": "Critical", "cost": 245000, "color": (80,160,20),   "parts": ["Total Front", "Frame Bent", "Engine"]},
    {"id": 8,  "speed": 85,  "damage": "Critical", "cost": 310000, "color": (40,170,40),   "parts": ["Full Body", "Roof", "All Systems"]},
    {"id": 9,  "speed": 100, "damage": "Total Loss","cost": 420000, "color": (20,160,100),  "parts": ["Vehicle Total Loss"]},
    {"id": 10, "speed": 120, "damage": "Total Loss","cost": 580000, "color": (10,140,160),  "parts": ["Vehicle Total Loss + Liability"]},
]

SEVERITY_COLORS = {
    "Minor":      ((255, 200, 50),  (220, 160, 30)),
    "Moderate":   ((255, 140, 20),  (200, 100, 10)),
    "Severe":     ((220, 60, 20),   (180, 30, 10)),
    "Critical":   ((180, 20, 20),   (140, 10, 10)),
    "Total Loss": ((120, 10, 10),   (80,  5,  5)),
}

def draw_car_body(draw, cx, cy, w, h, body_color=(60, 80, 120)):
    """Draw a simplified top-down car silhouette."""
    # Main body
    draw.rounded_rectangle([cx-w//2, cy-h//2, cx+w//2, cy+h//2], radius=30, fill=body_color, outline=(30,30,30), width=3)
    # Roof (lighter)
    roof_color = tuple(min(c+40, 255) for c in body_color)
    draw.rounded_rectangle([cx-w//3, cy-h//3, cx+w//3, cy+h//3], radius=20, fill=roof_color)
    # Windshield
    draw.polygon([cx-w//3+10, cy-h//3, cx+w//3-10, cy-h//3,
                  cx+w//4, cy-h//5, cx-w//4, cy-h//5], fill=(150,200,230,180))
    # Rear window
    draw.polygon([cx-w//3+10, cy+h//3, cx+w//3-10, cy+h//3,
                  cx+w//4, cy+h//5, cx-w//4, cy+h//5], fill=(150,200,230,180))
    # Wheels
    wheel_color = (25, 25, 25)
    for wx, wy in [(cx-w//2+15, cy-h//2+20), (cx+w//2-15, cy-h//2+20),
                   (cx-w//2+15, cy+h//2-20), (cx+w//2-15, cy+h//2-20)]:
        draw.ellipse([wx-18, wy-18, wx+18, wy+18], fill=wheel_color, outline=(60,60,60), width=2)
        draw.ellipse([wx-8,  wy-8,  wx+8,  wy+8],  fill=(80,80,80))

def draw_damage_cracks(draw, cx, cy, severity, speed, damage_zone="front"):
    """Draw crack and deformation patterns based on severity."""
    rng = random.Random(speed * 42)

    num_cracks  = int(speed / 10) + 2
    max_len     = int(speed * 1.8)
    crack_color = (20, 20, 20)
    dent_color  = (40, 40, 40, 120)

    # Damage origin point
    if damage_zone == "front":
        ox, oy = cx, cy - 110
    else:
        ox, oy = cx + rng.randint(-60, 60), cy + rng.randint(-60, 60)

    # Radiating cracks
    for i in range(num_cracks):
        angle  = rng.uniform(0, 2 * math.pi)
        length = rng.randint(max_len // 3, max_len)
        ex = int(ox + length * math.cos(angle))
        ey = int(oy + length * math.sin(angle))
        width = rng.randint(1, max(2, speed // 25))
        draw.line([ox, oy, ex, ey], fill=crack_color, width=width)
        # Branch cracks
        if severity in ("Severe", "Critical", "Total Loss"):
            for _ in range(rng.randint(1, 3)):
                blen   = length // 3
                bangle = angle + rng.uniform(-1, 1)
                bx = int(ex + blen * math.cos(bangle))
                by = int(ey + blen * math.sin(bangle))
                draw.line([ex, ey, bx, by], fill=crack_color, width=1)

    # Dent circles
    num_dents = max(1, speed // 20)
    for _ in range(num_dents):
        dx = ox + rng.randint(-40, 40)
        dy = oy + rng.randint(-30, 30)
        dr = rng.randint(15, 35)
        draw.ellipse([dx-dr, dy-dr, dx+dr, dy+dr], outline=(30,30,30), width=2)

    # Impact point (bright flash for high speed)
    if speed > 60:
        flash_r = speed // 8
        draw.ellipse([ox-flash_r, oy-flash_r, ox+flash_r, oy+flash_r],
                     fill=(255, 220, 100), outline=(255,160,0), width=2)

def draw_debris(draw, cx, cy, speed, rng):
    """Scatter debris particles for high-speed incidents."""
    num = int(speed / 8)
    for _ in range(num):
        px = cx + rng.randint(-180, 180)
        py = cy + rng.randint(-140, 140)
        ps = rng.randint(2, 8)
        col = rng.choice([(80,80,80),(120,100,60),(200,180,100),(60,60,60)])
        draw.ellipse([px-ps, py-ps, px+ps, py+ps], fill=col)

def generate_incident_image(incident):
    W, H = 800, 600
    img  = Image.new("RGB", (W, H), (18, 20, 28))
    draw = ImageDraw.Draw(img, "RGBA")
    rng  = random.Random(incident["speed"] * 7 + incident["id"])

    # ── Background grid ───────────────────────────────────────────────────
    for x in range(0, W, 40):
        draw.line([(x,0),(x,H)], fill=(30,35,50), width=1)
    for y in range(0, H, 40):
        draw.line([(0,y),(W,y)], fill=(30,35,50), width=1)

    # ── Road surface ───────────────────────────────────────────────────────
    road_pts = [(0, H//2+60),(W, H//2+60),(W,H),(0,H)]
    draw.polygon(road_pts, fill=(35,38,45))
    # Road markings
    for x in range(50, W, 120):
        draw.rectangle([x, H//2+100, x+60, H//2+108], fill=(180,160,40))

    # ── Speed streaks (motion blur effect) ────────────────────────────────
    if incident["speed"] > 40:
        for _ in range(incident["speed"] // 5):
            sx = rng.randint(0, W)
            sy = rng.randint(H//4, H//2+50)
            sl = rng.randint(30, 120)
            alpha = rng.randint(30, 90)
            draw.line([(sx,sy),(sx+sl,sy)], fill=(200,220,255,alpha), width=1)

    # ── Car body ───────────────────────────────────────────────────────────
    cx, cy  = W//2, H//2 - 30
    car_col = (50+rng.randint(0,30), 70+rng.randint(0,20), 110+rng.randint(0,40))
    draw_car_body(draw, cx, cy, 160, 260, car_col)

    # ── Damage visualization ───────────────────────────────────────────────
    draw_damage_cracks(draw, cx, cy, incident["damage"], incident["speed"])

    # ── Debris ────────────────────────────────────────────────────────────
    if incident["speed"] > 30:
        draw_debris(draw, cx, cy, incident["speed"], rng)

    # ── Smoke / fire effect for high speed ────────────────────────────────
    if incident["speed"] > 70:
        for i in range(20):
            sx = cx + rng.randint(-40, 40)
            sy = cy - 120 - i * 8
            sr = rng.randint(15, 35)
            alpha = max(20, 180 - i * 9)
            smoke_col = (60+i*3, 60+i*3, 60+i*3, alpha)
            draw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr], fill=smoke_col)
        # Fire
        for i in range(12):
            fx = cx + rng.randint(-25, 25)
            fy = cy - 100 - i * 5
            fr = rng.randint(8, 20)
            fire_col = (255, rng.randint(60,180), 0, 200-i*10)
            draw.ellipse([fx-fr, fy-fr, fx+fr, fy+fr], fill=fire_col)

    # ── Severity badge ─────────────────────────────────────────────────────
    sev   = incident["damage"]
    c1,c2 = SEVERITY_COLORS[sev]
    bx, by, bw, bh = 30, 30, 200, 50
    draw.rounded_rectangle([bx,by,bx+bw,by+bh], radius=10, fill=c1, outline=c2, width=2)

    # ── Info overlay panel ─────────────────────────────────────────────────
    panel_x, panel_y = W-240, 20
    draw.rounded_rectangle([panel_x, panel_y, W-20, panel_y+160],
                            radius=12, fill=(10,12,20,210), outline=(60,80,120), width=1)

    # ── Apply slight blur for realism ──────────────────────────────────────
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    draw = ImageDraw.Draw(img)

    # ── Text labels ────────────────────────────────────────────────────────
    try:
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_med   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font_big = font_med = font_small = ImageFont.load_default()

    # Severity badge text
    draw.text((bx+16, by+14), f"⚠  {sev.upper()}", fill=(255,255,255), font=font_big)

    # Info panel text
    px = panel_x + 16
    draw.text((px, panel_y+12),  f"Incident #{incident['id']:02d}",            fill=(180,200,255), font=font_med)
    draw.text((px, panel_y+35),  f"Speed   : {incident['speed']} km/h",        fill=(220,220,220), font=font_small)
    draw.text((px, panel_y+55),  f"Damage  : {incident['damage']}",            fill=(220,220,220), font=font_small)
    draw.text((px, panel_y+75),  f"Est Cost: ₹{incident['cost']:,}",           fill=(100,255,150), font=font_small)
    draw.text((px, panel_y+100), "Parts affected:",                             fill=(160,180,220), font=font_small)
    for i, part in enumerate(incident["parts"][:3]):
        draw.text((px+8, panel_y+118+i*16), f"• {part}",                       fill=(200,200,200), font=font_small)

    # Speed gauge at bottom
    gauge_x, gauge_y = 30, H - 80
    draw.rounded_rectangle([gauge_x, gauge_y, gauge_x+300, gauge_y+40],
                            radius=8, fill=(20,22,32), outline=(50,60,90), width=1)
    speed_pct = min(incident["speed"] / 120, 1.0)
    bar_w     = int(296 * speed_pct)
    s1, s2    = SEVERITY_COLORS[sev]
    if bar_w > 4:
        draw.rounded_rectangle([gauge_x+2, gauge_y+2, gauge_x+2+bar_w, gauge_y+38],
                                radius=6, fill=s1)
    draw.text((gauge_x+10, gauge_y+10), f"Impact Speed: {incident['speed']} km/h",
              fill=(255,255,255), font=font_med)

    # Footer
    draw.text((W//2-120, H-28), "AI Car Repair Cost Predictor  |  Automotive ML",
              fill=(60,80,100), font=font_small)

    path = os.path.join(OUTPUT_DIR, f"incident_{incident['id']:02d}_speed_{incident['speed']}kmh.png")
    img.save(path, "PNG", quality=95)
    print(f"  ✓ Saved: {path}")
    return path

if __name__ == "__main__":
    print("\n🚗  Generating 10 car incident images...\n")
    paths = []
    for inc in INCIDENTS:
        p = generate_incident_image(inc)
        paths.append(p)
    print(f"\n✅  All {len(paths)} images saved to {OUTPUT_DIR}/")
