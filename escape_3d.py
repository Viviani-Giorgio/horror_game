import math
import os
import random
from collections import deque
import pygame


WIDTH, HEIGHT = 1280, 720
FPS = 60

TILE_SIZE = 120
MAZE_ROWS = 31
MAZE_COLS = 31

FOV = math.radians(70)
HALF_FOV = FOV / 2
NUM_RAYS = 320
DELTA_ANGLE = FOV / NUM_RAYS
SCALE = WIDTH // NUM_RAYS
MAX_DEPTH = TILE_SIZE * max(MAZE_ROWS, MAZE_COLS)
PROJ_COEFF = (WIDTH / 2) / math.tan(HALF_FOV)
MAX_DDA_STEPS = MAZE_ROWS * MAZE_COLS

PLAYER_RADIUS = 18
PLAYER_SPEED = 240.0
SPRINT_MULT = 1.65
ROT_SPEED = 0.0022

STAMINA_MAX = 100.0
STAMINA_DRAIN = 45.0
STAMINA_REGEN = 30.0

BATTERY_MAX = 150.0
BATTERY_DRAIN = 1.75
BATTERY_PICKUP = 30.0
LIGHT_MIN = 90
LIGHT_MAX = 520

KEYS_REQUIRED = 3
MONSTER_RADIUS = 12
MONSTER_BASE_SPEED = 115.0
MONSTER_MAX_MULT = 1.75
MONSTER_PATH_INTERVAL = 0.33

BOOST_TIME = 6.0
START_GRACE_TIME = 7.0
WORLD_UNITS_PER_METER = 64.0
HEARTBEAT_TRIGGER_METERS = 20.0
SOUND_DIR = "sounds"

DIFFICULTY_PRESETS = {
    "Facile": {
        "monster_speed_mult": 0.82,
        "monster_max_mult": 0.86,
        "grace_time": 9.0,
        "battery_drain_mult": 0.8,
        "keys": 2,
    },
    "Normale": {
        "monster_speed_mult": 1.0,
        "monster_max_mult": 1.0,
        "grace_time": 7.0,
        "battery_drain_mult": 1.0,
        "keys": 3,
    },
    "Difficile": {
        "monster_speed_mult": 1.2,
        "monster_max_mult": 1.15,
        "grace_time": 5.0,
        "battery_drain_mult": 1.18,
        "keys": 4,
    },
}


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Escape3DGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("The Monster")

        self.fullscreen = False
        self.display = self.create_display()
        self.screen = pygame.Surface((WIDTH, HEIGHT)).convert()
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 22)
        self.big_font = pygame.font.SysFont("consolas", 42, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 17)

        self.running = True
        self.paused = False
        self.show_minimap = False
        self.mouse_captured = False
        self.scene = "menu"
        self.menu_enter_time = pygame.time.get_ticks()

        self.menu_index = 0
        self.menu_difficulties = list(DIFFICULTY_PRESETS.keys())
        self.menu_difficulty_index = self.menu_difficulties.index("Normale")
        self.menu_music_volume = 70
        self.menu_sfx_volume = 80
        self.menu_minimap_default = False

        self.monster_sprite = self.load_external_sprite(
            os.path.join("img", "ck.jpeg"),
            (320, 320),
            (210, 70, 70),
            "CK",
        )
        self.key_sprite = self.make_icon_sprite((130, 130), (255, 210, 65), "K", "circle")
        self.battery_sprite = self.make_icon_sprite((130, 130), (80, 220, 255), "B", "square")
        self.boost_sprite = self.make_icon_sprite((130, 130), (120, 255, 120), "S", "diamond")
        self.exit_locked_sprite = self.make_exit_sprite((130, 170), locked=True)
        self.exit_open_sprite = self.make_exit_sprite((130, 170), locked=False)

        self.zbuffer = [MAX_DEPTH] * NUM_RAYS
        self.sprite_cache = {}
        self.flashlight_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.minimap_base = None
        self.menu_background = self.build_menu_background()
        self.minimap_scale = 5
        self.status_text = ""
        self.status_timer = 0.0
        self.last_world_signature = None
        self.music_volume = self.menu_music_volume / 100.0
        self.sfx_volume = self.menu_sfx_volume / 100.0
        self.current_music_path = None
        self.heartbeat_channel = None
        self.heartbeat_sound = None
        self.menu_music_path = None
        self.game_music_path = None
        self.audio_enabled = False
        self.missing_audio = []

        self.setup_audio()
        self.reset_world(capture_mouse=False)
        self.set_mouse_capture(False)

    def create_display(self):
        if self.fullscreen:
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            mode_size = (0, 0)
        else:
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
            mode_size = (WIDTH, HEIGHT)

        try:
            display = pygame.display.set_mode(mode_size, flags, vsync=1)
        except TypeError:
            display = pygame.display.set_mode(mode_size, flags)

        self.update_viewport(display.get_size())
        return display

    def update_viewport(self, display_size):
        dw, dh = display_size
        scale = min(dw / WIDTH, dh / HEIGHT)
        view_w = max(1, int(WIDTH * scale))
        view_h = max(1, int(HEIGHT * scale))
        self.viewport_rect = pygame.Rect((dw - view_w) // 2, (dh - view_h) // 2, view_w, view_h)
        self.scale_buffer = None
        if view_w != WIDTH or view_h != HEIGHT:
            self.scale_buffer = pygame.Surface((view_w, view_h)).convert()

    def present_frame(self):
        if self.scale_buffer is not None:
            pygame.transform.scale(self.screen, self.viewport_rect.size, self.scale_buffer)
            self.display.fill((0, 0, 0))
            self.display.blit(self.scale_buffer, self.viewport_rect.topleft)
        else:
            self.display.blit(self.screen, (0, 0))
        pygame.display.flip()

    def build_menu_background(self):
        surface = pygame.Surface((WIDTH, HEIGHT))
        top = (20, 5, 6)
        bottom = (4, 1, 1)

        for y in range(HEIGHT):
            t = y / (HEIGHT - 1)
            r = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            b = int(top[2] * (1 - t) + bottom[2] * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (WIDTH, y))

        blood_haze = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(blood_haze, (210, 16, 24, 52), (WIDTH // 2, HEIGHT // 2 - 40), 360)
        pygame.draw.circle(blood_haze, (160, 10, 18, 44), (210, 140), 220)
        pygame.draw.circle(blood_haze, (160, 10, 18, 38), (1100, 560), 260)
        surface.blit(blood_haze, (0, 0))

        accent = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for x in range(-240, WIDTH + 260, 120):
            pygame.draw.line(accent, (214, 164, 72, 20), (x, 0), (x + 190, HEIGHT), 2)
        surface.blit(accent, (0, 0))

        vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.ellipse(vignette, (0, 0, 0, 118), (-220, -140, WIDTH + 440, HEIGHT + 280), 230)
        surface.blit(vignette, (0, 0))

        grain = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for _ in range(280):
            x = random.randrange(0, WIDTH)
            y = random.randrange(0, HEIGHT)
            grain.set_at((x, y), (225, 185, 110, 8))
        surface.blit(grain, (0, 0))
        return surface

    def find_sound_path(self, *base_names):
        if not os.path.isdir(SOUND_DIR):
            return None

        for base_name in base_names:
            if not base_name:
                continue

            direct_path = os.path.join(SOUND_DIR, base_name)
            if os.path.isfile(direct_path):
                return direct_path

            for ext in (".mp3", ".wav", ".ogg"):
                path = os.path.join(SOUND_DIR, f"{base_name}{ext}")
                if os.path.isfile(path):
                    return path

            for filename in os.listdir(SOUND_DIR):
                root, _ = os.path.splitext(filename)
                if root.lower() == base_name.lower():
                    path = os.path.join(SOUND_DIR, filename)
                    if os.path.isfile(path):
                        return path

        return None

    def setup_audio(self):
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            self.audio_enabled = True
        except pygame.error:
            self.audio_enabled = False
            return

        self.menu_music_path = self.find_sound_path("limited", "random")
        self.game_music_path = self.find_sound_path("sub")
        heartbeat_path = self.find_sound_path("hrtb")

        if self.menu_music_path is None:
            self.missing_audio.append("limited/random")
        if self.game_music_path is None:
            self.missing_audio.append("sub")
        if heartbeat_path is None:
            self.missing_audio.append("hrtb")

        if heartbeat_path is not None:
            try:
                self.heartbeat_sound = pygame.mixer.Sound(heartbeat_path)
                self.heartbeat_channel = pygame.mixer.Channel(1)
            except pygame.error:
                self.heartbeat_sound = None
                self.heartbeat_channel = None

        self.apply_audio_volumes()

    def apply_audio_volumes(self):
        self.music_volume = clamp(self.menu_music_volume / 100.0, 0.0, 1.0)
        self.sfx_volume = clamp(self.menu_sfx_volume / 100.0, 0.0, 1.0)

        if not self.audio_enabled:
            return

        pygame.mixer.music.set_volume(self.music_volume)
        if self.heartbeat_channel is not None:
            self.heartbeat_channel.set_volume(self.sfx_volume)

    def set_music_track(self, path):
        if not self.audio_enabled:
            return
        if path == self.current_music_path:
            return

        pygame.mixer.music.stop()
        self.current_music_path = None

        if path is None:
            return

        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(self.music_volume)
            pygame.mixer.music.play(-1, fade_ms=300)
            self.current_music_path = path
        except pygame.error:
            self.current_music_path = None

    def stop_heartbeat(self):
        if self.heartbeat_channel is not None and self.heartbeat_channel.get_busy():
            self.heartbeat_channel.stop()

    def update_heartbeat_audio(self):
        if self.heartbeat_channel is None or self.heartbeat_sound is None:
            return
        if self.scene != "game" or self.state != "playing" or self.paused:
            self.stop_heartbeat()
            return

        distance_units = math.hypot(self.player_x - self.monster_x, self.player_y - self.monster_y)
        distance_meters = distance_units / WORLD_UNITS_PER_METER

        if distance_meters <= HEARTBEAT_TRIGGER_METERS:
            proximity = 1.0 - (distance_meters / HEARTBEAT_TRIGGER_METERS)
            heartbeat_volume = clamp(0.2 + proximity * 0.8, 0.0, 1.0) * self.sfx_volume
            if not self.heartbeat_channel.get_busy():
                self.heartbeat_channel.play(self.heartbeat_sound, loops=-1, fade_ms=150)
            self.heartbeat_channel.set_volume(heartbeat_volume)
        else:
            self.stop_heartbeat()

    def update_audio_state(self):
        if not self.audio_enabled:
            return

        if self.scene == "menu":
            self.set_music_track(self.menu_music_path)
            self.stop_heartbeat()
            return

        if self.state == "playing" and not self.paused:
            self.set_music_track(self.game_music_path)
        else:
            self.set_music_track(None)

        self.update_heartbeat_audio()

    def load_external_sprite(self, path, size, color, label):
        try:
            image = pygame.image.load(path).convert_alpha()
            return pygame.transform.smoothscale(image, size)
        except Exception:
            fallback = pygame.Surface(size, pygame.SRCALPHA)
            pygame.draw.circle(
                fallback,
                color,
                (size[0] // 2, size[1] // 2),
                min(size[0], size[1]) // 2 - 6,
            )
            txt = self.big_font.render(label, True, (20, 20, 20))
            fallback.blit(txt, txt.get_rect(center=(size[0] // 2, size[1] // 2)))
            return fallback

    def make_icon_sprite(self, size, color, label, shape):
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 8

        if shape == "square":
            pygame.draw.rect(surf, color, (16, 16, w - 32, h - 32), border_radius=12)
            pygame.draw.rect(surf, (15, 15, 15), (16, 16, w - 32, h - 32), 4, border_radius=12)
        elif shape == "diamond":
            points = [(center[0], 10), (w - 10, center[1]), (center[0], h - 10), (10, center[1])]
            pygame.draw.polygon(surf, color, points)
            pygame.draw.polygon(surf, (15, 15, 15), points, 4)
        else:
            pygame.draw.circle(surf, color, center, radius)
            pygame.draw.circle(surf, (15, 15, 15), center, radius, 4)

        txt = self.big_font.render(label, True, (20, 20, 20))
        surf.blit(txt, txt.get_rect(center=center))
        return surf

    def make_exit_sprite(self, size, locked):
        surf = pygame.Surface(size, pygame.SRCALPHA)
        w, h = size
        door_color = (60, 180, 80) if not locked else (200, 70, 70)
        pygame.draw.rect(surf, door_color, (10, 10, w - 20, h - 20), border_radius=8)
        pygame.draw.rect(surf, (20, 20, 20), (10, 10, w - 20, h - 20), 5, border_radius=8)
        pygame.draw.circle(surf, (250, 230, 120), (w - 24, h // 2), 8)
        return surf

    def get_scaled_sprite(self, image, sprite_size):
        size = max(2, int(sprite_size))
        quantized = max(2, (size // 4) * 4)
        key = (id(image), quantized)
        cached = self.sprite_cache.get(key)
        if cached is None:
            cached = pygame.transform.smoothscale(image, (quantized, quantized))
            self.sprite_cache[key] = cached
            if len(self.sprite_cache) > 300:
                self.sprite_cache.pop(next(iter(self.sprite_cache)))
        return cached

    def build_minimap_base(self):
        map_w = self.map_cols * self.minimap_scale
        map_h = self.map_rows * self.minimap_scale
        surf = pygame.Surface((map_w, map_h))

        for y in range(self.map_rows):
            for x in range(self.map_cols):
                color = (38, 38, 38) if self.map_grid[y][x] == 1 else (92, 92, 92)
                pygame.draw.rect(
                    surf,
                    color,
                    (x * self.minimap_scale, y * self.minimap_scale, self.minimap_scale, self.minimap_scale),
                )

        self.minimap_base = surf

    def set_status(self, text, duration=2.0):
        self.status_text = text
        self.status_timer = duration

    def set_mouse_capture(self, capture):
        self.mouse_captured = capture
        try:
            pygame.event.set_grab(capture)
            pygame.mouse.set_visible(not capture)
        except Exception:
            pass
        if capture:
            pygame.mouse.get_rel()

    def current_difficulty_name(self):
        return self.menu_difficulties[self.menu_difficulty_index]

    def enter_menu(self):
        self.scene = "menu"
        self.paused = False
        self.set_mouse_capture(False)
        self.menu_enter_time = pygame.time.get_ticks()

    def menu_option_count(self):
        return 7

    def menu_adjust(self, direction):
        if self.menu_index == 1:
            self.menu_difficulty_index = (self.menu_difficulty_index + direction) % len(self.menu_difficulties)
        elif self.menu_index == 2:
            self.menu_music_volume = int(clamp(self.menu_music_volume + direction * 5, 0, 100))
            self.apply_audio_volumes()
        elif self.menu_index == 3:
            self.menu_sfx_volume = int(clamp(self.menu_sfx_volume + direction * 5, 0, 100))
            self.apply_audio_volumes()
        elif self.menu_index == 4:
            self.menu_minimap_default = not self.menu_minimap_default
        elif self.menu_index == 5:
            self.toggle_fullscreen()

    def start_game_from_menu(self):
        self.scene = "game"
        self.show_minimap = self.menu_minimap_default if self.current_difficulty_name() != "Difficile" else False
        self.reset_world(capture_mouse=True)
        self.set_status("Sopravvivi e raggiungi l'uscita", duration=2.5)

    def activate_menu_item(self):
        if self.menu_index == 0:
            self.start_game_from_menu()
        elif self.menu_index == 6:
            self.running = False
        else:
            self.menu_adjust(1)

    def generate_maze(self, rows, cols):
        if rows % 2 == 0:
            rows += 1
        if cols % 2 == 0:
            cols += 1

        grid = [[1 for _ in range(cols)] for _ in range(rows)]
        start_x = 1 + 2 * self.rng.randrange(max(1, (cols - 1) // 2))
        start_y = 1 + 2 * self.rng.randrange(max(1, (rows - 1) // 2))
        stack = [(start_x, start_y)]
        grid[start_y][start_x] = 0

        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        while stack:
            cx, cy = stack[-1]
            self.rng.shuffle(dirs)
            carved = False

            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < cols - 1 and 1 <= ny < rows - 1 and grid[ny][nx] == 1:
                    grid[cy + dy // 2][cx + dx // 2] = 0
                    grid[ny][nx] = 0
                    stack.append((nx, ny))
                    carved = True
                    break

            if not carved:
                stack.pop()

        # Collega la cella iniziale (1,1) al labirinto senza creare "stanzoni".
        if grid[1][1] == 1:
            frontier = deque([(1, 1)])
            seen = {(1, 1)}
            while frontier:
                cx, cy = frontier.popleft()
                if grid[cy][cx] == 0:
                    px, py = 1, 1
                    while (px, py) != (cx, cy):
                        if px < cx:
                            px += 1
                        elif px > cx:
                            px -= 1
                        elif py < cy:
                            py += 1
                        else:
                            py -= 1
                        grid[py][px] = 0
                    break

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 1 <= nx < cols - 1 and 1 <= ny < rows - 1 and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        frontier.append((nx, ny))

        return grid

    def grid_signature(self, grid):
        return bytes(cell for row in grid for cell in row)

    def compute_distance_map(self, start):
        queue = deque([start])
        dist = {start: 0}

        while queue:
            cell = queue.popleft()
            for nxt in self.neighbors(cell):
                if nxt not in dist:
                    dist[nxt] = dist[cell] + 1
                    queue.append(nxt)

        return dist

    def world_to_cell(self, x, y):
        return int(x // TILE_SIZE), int(y // TILE_SIZE)

    def cell_to_world_center(self, cell):
        cx, cy = cell
        return (cx + 0.5) * TILE_SIZE, (cy + 0.5) * TILE_SIZE

    def is_walkable_cell(self, cx, cy):
        if 0 <= cy < self.map_rows and 0 <= cx < self.map_cols:
            return self.map_grid[cy][cx] == 0
        return False

    def is_wall_world(self, x, y):
        cx, cy = self.world_to_cell(x, y)
        return not self.is_walkable_cell(cx, cy)

    def can_move(self, x, y, radius):
        checks = [
            (-radius, -radius),
            (radius, -radius),
            (-radius, radius),
            (radius, radius),
            (0, 0),
        ]
        for ox, oy in checks:
            if self.is_wall_world(x + ox, y + oy):
                return False
        return True

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, cell):
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if self.is_walkable_cell(nx, ny):
                yield (nx, ny)

    def farthest_cell(self, start, avoid=None):
        avoid = avoid or set()
        queue = deque([start])
        seen = {start}
        farthest = start

        while queue:
            current = queue.popleft()
            if current not in avoid:
                farthest = current

            for nxt in self.neighbors(current):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        return farthest

    def pick_random_cells(self, count, blocked, min_distance_start=0):
        floors = []
        for y, row in enumerate(self.map_grid):
            for x, val in enumerate(row):
                cell = (x, y)
                if val == 0 and cell not in blocked:
                    floors.append(cell)

        self.rng.shuffle(floors)
        selected = []

        for cell in floors:
            if len(selected) >= count:
                break
            if self.manhattan(cell, self.start_cell) < min_distance_start:
                continue
            if any(self.manhattan(cell, other) < 4 for other in selected):
                continue
            selected.append(cell)
            blocked.add(cell)

        if len(selected) < count:
            for cell in floors:
                if len(selected) >= count:
                    break
                if cell in blocked:
                    continue
                if self.manhattan(cell, self.start_cell) < max(2, min_distance_start // 2):
                    continue
                selected.append(cell)
                blocked.add(cell)

        return selected

    def find_path(self, start, goal):
        if start == goal:
            return [start]
        if not self.is_walkable_cell(start[0], start[1]) or not self.is_walkable_cell(goal[0], goal[1]):
            return [start]

        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break

            for nxt in self.neighbors(current):
                if nxt not in came_from:
                    came_from[nxt] = current
                    queue.append(nxt)

        if goal not in came_from:
            return [start]

        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    def has_line_of_sight(self, cell_a, cell_b):
        x0, y0 = cell_a
        x1, y1 = cell_b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not self.is_walkable_cell(x0, y0):
                return False
            if (x0, y0) == (x1, y1):
                return True
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def reset_world(self, capture_mouse=True):
        self.rng = random.Random(int.from_bytes(os.urandom(16), "big"))
        for attempt in range(8):
            self.map_grid = self.generate_maze(MAZE_ROWS, MAZE_COLS)
            signature = self.grid_signature(self.map_grid)
            if signature != self.last_world_signature or attempt == 7:
                self.last_world_signature = signature
                break
            self.rng.seed(int.from_bytes(os.urandom(16), "big"))

        self.map_rows = len(self.map_grid)
        self.map_cols = len(self.map_grid[0])

        self.world_w = self.map_cols * TILE_SIZE
        self.world_h = self.map_rows * TILE_SIZE

        self.start_cell = (1, 1)
        self.player_x, self.player_y = self.cell_to_world_center(self.start_cell)
        self.player_angle = 0.0

        self.stamina = STAMINA_MAX
        self.battery = BATTERY_MAX
        self.flashlight_on = True
        self.speed_boost_time = 0.0

        self.elapsed = 0.0
        self.state = "playing"
        self.paused = False
        self.status_text = ""
        self.status_timer = 0.0

        difficulty = DIFFICULTY_PRESETS[self.current_difficulty_name()]
        self.monster_speed_mult = difficulty["monster_speed_mult"]
        self.monster_max_mult = difficulty["monster_max_mult"]
        self.battery_drain = BATTERY_DRAIN * difficulty["battery_drain_mult"]
        self.keys_target = max(1, int(difficulty["keys"]))
        self.monster_grace_time = difficulty["grace_time"]

        dist_from_start = self.compute_distance_map(self.start_cell)
        max_dist = max(dist_from_start.values())

        exit_threshold = max(12, int(max_dist * 0.72))
        exit_candidates = [
            cell for cell, distance in dist_from_start.items() if distance >= exit_threshold and cell != self.start_cell
        ]
        if not exit_candidates:
            exit_candidates = [cell for cell, distance in dist_from_start.items() if distance == max_dist]
        self.exit_cell = self.rng.choice(exit_candidates)

        monster_threshold = max(16, int(max_dist * 0.52))
        monster_candidates = [
            cell
            for cell, distance in dist_from_start.items()
            if distance >= monster_threshold
            and cell not in {self.start_cell, self.exit_cell}
            and self.manhattan(cell, self.exit_cell) >= 8
        ]
        if not monster_candidates:
            monster_candidates = [cell for cell in dist_from_start if cell not in {self.start_cell, self.exit_cell}]
        monster_cell = self.rng.choice(monster_candidates)

        self.monster_x, self.monster_y = self.cell_to_world_center(monster_cell)
        self.monster_path = [monster_cell]
        self.path_timer = 0.0
        self.monster_dir = (1.0, 0.0)
        self.monster_grace_timer = self.monster_grace_time

        blocked = {self.start_cell, self.exit_cell, monster_cell}
        self.key_cells = self.pick_random_cells(self.keys_target, blocked, min_distance_start=8)
        self.battery_cells = self.pick_random_cells(6, blocked, min_distance_start=5)
        self.boost_cells = self.pick_random_cells(3, blocked, min_distance_start=7)

        self.keys_required = len(self.key_cells)
        self.keys_collected = 0
        self.exit_unlocked = self.keys_required == 0

        self.sprite_cache.clear()
        self.build_minimap_base()
        self.set_mouse_capture(capture_mouse)

    def update_player(self, dt):
        if self.mouse_captured:
            rel_x = pygame.mouse.get_rel()[0]
            self.player_angle = (self.player_angle + rel_x * ROT_SPEED) % (2 * math.pi)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player_angle = (self.player_angle - 2.6 * ROT_SPEED * 60 * dt) % (2 * math.pi)
        if keys[pygame.K_RIGHT]:
            self.player_angle = (self.player_angle + 2.6 * ROT_SPEED * 60 * dt) % (2 * math.pi)

        sin_a = math.sin(self.player_angle)
        cos_a = math.cos(self.player_angle)

        dir_x = 0.0
        dir_y = 0.0

        if keys[pygame.K_w]:
            dir_x += cos_a
            dir_y += sin_a
        if keys[pygame.K_s]:
            dir_x -= cos_a
            dir_y -= sin_a
        if keys[pygame.K_a]:
            dir_x += sin_a
            dir_y -= cos_a
        if keys[pygame.K_d]:
            dir_x -= sin_a
            dir_y += cos_a

        move_len = math.hypot(dir_x, dir_y)
        if move_len > 0:
            dir_x /= move_len
            dir_y /= move_len

        sprinting = (
            move_len > 0
            and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            and self.stamina > 0.5
        )

        speed = PLAYER_SPEED
        if self.speed_boost_time > 0:
            speed *= 1.2

        if sprinting:
            speed *= SPRINT_MULT
            self.stamina -= STAMINA_DRAIN * dt
        else:
            self.stamina += STAMINA_REGEN * dt

        self.stamina = clamp(self.stamina, 0.0, STAMINA_MAX)

        step = speed * dt
        new_x = self.player_x + dir_x * step
        new_y = self.player_y + dir_y * step

        if self.can_move(new_x, self.player_y, PLAYER_RADIUS):
            self.player_x = new_x
        if self.can_move(self.player_x, new_y, PLAYER_RADIUS):
            self.player_y = new_y

        if self.speed_boost_time > 0:
            self.speed_boost_time = max(0.0, self.speed_boost_time - dt)

        if self.flashlight_on:
            self.battery -= self.battery_drain * dt
            if self.battery <= 0:
                self.battery = 0.0
                self.flashlight_on = False
                self.set_status("Batteria scarica")
        else:
            self.battery = clamp(self.battery, 0.0, BATTERY_MAX)

    def collect_pickups(self):
        pickup_dist_sq = (TILE_SIZE * 0.35) ** 2

        for cell in self.key_cells[:]:
            cx, cy = self.cell_to_world_center(cell)
            if (self.player_x - cx) ** 2 + (self.player_y - cy) ** 2 <= pickup_dist_sq:
                self.key_cells.remove(cell)
                self.keys_collected += 1
                self.set_status(f"Chiave presa: {self.keys_collected}/{self.keys_required}")
                if self.keys_collected >= self.keys_required:
                    self.exit_unlocked = True
                    self.set_status("Uscita sbloccata!")

        for cell in self.battery_cells[:]:
            cx, cy = self.cell_to_world_center(cell)
            if (self.player_x - cx) ** 2 + (self.player_y - cy) ** 2 <= pickup_dist_sq:
                self.battery_cells.remove(cell)
                self.battery = clamp(self.battery + BATTERY_PICKUP, 0.0, BATTERY_MAX)
                self.set_status("Batteria ricaricata")

        for cell in self.boost_cells[:]:
            cx, cy = self.cell_to_world_center(cell)
            if (self.player_x - cx) ** 2 + (self.player_y - cy) ** 2 <= pickup_dist_sq:
                self.boost_cells.remove(cell)
                self.speed_boost_time = BOOST_TIME
                self.set_status("Boost velocita attivo")

    def update_monster(self, dt):
        monster_cell = self.world_to_cell(self.monster_x, self.monster_y)
        player_cell = self.world_to_cell(self.player_x, self.player_y)

        if self.monster_grace_timer > 0:
            self.monster_grace_timer = max(0.0, self.monster_grace_timer - dt)
            return

        self.path_timer -= dt
        if self.path_timer <= 0:
            self.monster_path = self.find_path(monster_cell, player_cell)
            self.path_timer = MONSTER_PATH_INTERVAL

        speed_mult = 1.0 + self.elapsed / 180.0
        speed_mult = min(speed_mult, MONSTER_MAX_MULT * self.monster_max_mult)
        monster_speed = MONSTER_BASE_SPEED * self.monster_speed_mult * speed_mult

        los = self.has_line_of_sight(monster_cell, player_cell)
        if los:
            target_x, target_y = self.player_x, self.player_y
            monster_speed *= 1.15
        else:
            target_cell = player_cell
            if len(self.monster_path) > 1:
                target_cell = self.monster_path[1]
            target_x, target_y = self.cell_to_world_center(target_cell)

        vx = target_x - self.monster_x
        vy = target_y - self.monster_y
        distance = math.hypot(vx, vy)
        if distance > 0.001:
            vx /= distance
            vy /= distance
            self.monster_dir = (vx, vy)

        step = monster_speed * dt
        move_x = self.monster_x + vx * step
        move_y = self.monster_y + vy * step

        if self.can_move(move_x, self.monster_y, MONSTER_RADIUS):
            self.monster_x = move_x
        if self.can_move(self.monster_x, move_y, MONSTER_RADIUS):
            self.monster_y = move_y

        if math.hypot(self.player_x - self.monster_x, self.player_y - self.monster_y) < (
            PLAYER_RADIUS + MONSTER_RADIUS + 6
        ):
            self.state = "lose"
            self.set_mouse_capture(False)

    def check_goal(self):
        if not self.exit_unlocked:
            return
        ex, ey = self.cell_to_world_center(self.exit_cell)
        if math.hypot(self.player_x - ex, self.player_y - ey) < TILE_SIZE * 0.35:
            self.state = "win"
            self.set_mouse_capture(False)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if self.scene == "menu":
                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.menu_index = (self.menu_index - 1) % self.menu_option_count()
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.menu_index = (self.menu_index + 1) % self.menu_option_count()
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.menu_adjust(-1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.menu_adjust(1)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.activate_menu_item()
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                continue

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.state == "playing" and not self.paused and not self.mouse_captured:
                    self.set_mouse_capture(True)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == "playing":
                        self.paused = not self.paused
                        self.set_mouse_capture(not self.paused)
                    else:
                        self.enter_menu()
                elif event.key == pygame.K_TAB and self.state == "playing":
                    if self.current_difficulty_name() == "Difficile":
                        self.set_status("Minimappa corrotta", 2.0)
                    else:
                        self.show_minimap = not self.show_minimap
                elif event.key == pygame.K_f and self.state == "playing":
                    if self.battery > 0 or self.flashlight_on:
                        self.flashlight_on = not self.flashlight_on
                elif event.key == pygame.K_r:
                    self.reset_world(capture_mouse=True)
                elif event.key == pygame.K_q and (self.paused or self.state in ("win", "lose")):
                    self.enter_menu()
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.display = self.create_display()
        self.set_mouse_capture(self.mouse_captured)

    def raycast_walls(self):
        self.screen.fill((24, 24, 30), (0, 0, WIDTH, HEIGHT // 2))
        self.screen.fill((34, 34, 34), (0, HEIGHT // 2, WIDTH, HEIGHT // 2))

        zbuffer = [MAX_DEPTH] * NUM_RAYS
        ray_angle = self.player_angle - HALF_FOV
        px = self.player_x
        py = self.player_y

        for ray in range(NUM_RAYS):
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)

            map_x = int(px // TILE_SIZE)
            map_y = int(py // TILE_SIZE)

            if dir_x < 0:
                step_x = -1
                next_grid_x = map_x * TILE_SIZE
                side_dist_x = (px - next_grid_x) / -dir_x
                delta_dist_x = TILE_SIZE / -dir_x
            elif dir_x > 0:
                step_x = 1
                next_grid_x = (map_x + 1) * TILE_SIZE
                side_dist_x = (next_grid_x - px) / dir_x
                delta_dist_x = TILE_SIZE / dir_x
            else:
                step_x = 0
                side_dist_x = float("inf")
                delta_dist_x = float("inf")

            if dir_y < 0:
                step_y = -1
                next_grid_y = map_y * TILE_SIZE
                side_dist_y = (py - next_grid_y) / -dir_y
                delta_dist_y = TILE_SIZE / -dir_y
            elif dir_y > 0:
                step_y = 1
                next_grid_y = (map_y + 1) * TILE_SIZE
                side_dist_y = (next_grid_y - py) / dir_y
                delta_dist_y = TILE_SIZE / dir_y
            else:
                step_y = 0
                side_dist_y = float("inf")
                delta_dist_y = float("inf")

            depth = MAX_DEPTH
            hit_side = 0

            for _ in range(MAX_DDA_STEPS):
                if side_dist_x < side_dist_y:
                    map_x += step_x
                    depth = side_dist_x
                    side_dist_x += delta_dist_x
                    hit_side = 0
                else:
                    map_y += step_y
                    depth = side_dist_y
                    side_dist_y += delta_dist_y
                    hit_side = 1

                if map_x < 0 or map_y < 0 or map_x >= self.map_cols or map_y >= self.map_rows:
                    depth = MAX_DEPTH
                    break
                if self.map_grid[map_y][map_x] == 1:
                    break

            if depth < MAX_DEPTH:
                corrected = depth * math.cos(self.player_angle - ray_angle)
                corrected = max(0.0001, corrected)
                proj_h = min(HEIGHT * 2, (TILE_SIZE * PROJ_COEFF) / corrected)

                shade_base = 255 - corrected * 0.08
                if hit_side == 1:
                    shade_base *= 0.84
                shade = int(clamp(shade_base, 24, 255))
                color = (shade, int(shade * 0.95), int(shade * 0.9))
                rect = (
                    ray * SCALE,
                    int(HEIGHT / 2 - proj_h / 2),
                    SCALE + 1,
                    int(proj_h),
                )
                pygame.draw.rect(self.screen, color, rect)
                zbuffer[ray] = corrected

            ray_angle += DELTA_ANGLE

        self.zbuffer = zbuffer

    def project_sprite(self, x, y, image, scale=1.0, y_shift=0.0):
        dx = x - self.player_x
        dy = y - self.player_y
        distance = math.hypot(dx, dy)
        if distance <= 1:
            return None

        theta = math.atan2(dy, dx)
        diff = normalize_angle(theta - self.player_angle)
        if abs(diff) > HALF_FOV + 0.5:
            return None

        corrected = distance * math.cos(diff)
        if corrected <= 0:
            return None

        screen_x = (diff + HALF_FOV) / FOV * WIDTH
        raw_size = int((TILE_SIZE * PROJ_COEFF / corrected) * scale)
        if raw_size < 2:
            return None
        sprite = self.get_scaled_sprite(image, raw_size)
        sprite_size = sprite.get_width()

        x_pos = int(screen_x - sprite_size / 2)
        y_pos = int(HEIGHT / 2 - sprite_size / 2 + y_shift * sprite_size)

        left_ray = max(0, int(x_pos / SCALE))
        right_ray = min(NUM_RAYS - 1, int((x_pos + sprite_size) / SCALE))
        if left_ray > right_ray:
            return None

        visible = False
        for ray in range(left_ray, right_ray + 1):
            if corrected < self.zbuffer[ray]:
                visible = True
                break
        if not visible:
            return None

        return corrected, sprite, x_pos, y_pos

    def draw_sprites(self):
        projected = []

        monster = self.project_sprite(
            self.monster_x,
            self.monster_y,
            self.monster_sprite,
            scale=1.35,
            y_shift=0.18,
        )
        if monster is not None:
            projected.append(monster)

        for cell in self.key_cells:
            x, y = self.cell_to_world_center(cell)
            p = self.project_sprite(x, y, self.key_sprite, scale=0.55, y_shift=0.3)
            if p is not None:
                projected.append(p)

        for cell in self.battery_cells:
            x, y = self.cell_to_world_center(cell)
            p = self.project_sprite(x, y, self.battery_sprite, scale=0.55, y_shift=0.3)
            if p is not None:
                projected.append(p)

        for cell in self.boost_cells:
            x, y = self.cell_to_world_center(cell)
            p = self.project_sprite(x, y, self.boost_sprite, scale=0.55, y_shift=0.3)
            if p is not None:
                projected.append(p)

        ex, ey = self.cell_to_world_center(self.exit_cell)
        exit_image = self.exit_open_sprite if self.exit_unlocked else self.exit_locked_sprite
        exit_proj = self.project_sprite(ex, ey, exit_image, scale=0.95, y_shift=0.25)
        if exit_proj is not None:
            projected.append(exit_proj)

        projected.sort(key=lambda item: item[0], reverse=True)
        for _, sprite, x_pos, y_pos in projected:
            self.screen.blit(sprite, (x_pos, y_pos))

    def draw_flashlight(self):
        overlay = self.flashlight_overlay
        center = (WIDTH // 2, HEIGHT // 2 + 42)

        def carve_soft_spot(base_alpha, radius, feather):
            overlay.fill((0, 0, 0, base_alpha))
            pygame.draw.circle(overlay, (0, 0, 0, 0), center, radius)

            # Bordo morbido: transizione continua, senza forme geometriche visibili.
            for i in range(max(1, feather)):
                t = i / max(1, feather - 1)
                alpha = int(base_alpha * (t ** 1.85))
                pygame.draw.circle(overlay, (0, 0, 0, alpha), center, radius + i, 1)

        if self.flashlight_on and self.battery > 0:
            ratio = self.battery / BATTERY_MAX
            radius = int(130 + 300 * ratio)
            feather = int(max(28, radius * 0.35))
            carve_soft_spot(base_alpha=214, radius=radius, feather=feather)
        else:
            # Torcia spenta: molto buio, ma non totalmente nero.
            carve_soft_spot(base_alpha=240, radius=72, feather=26)
        self.screen.blit(overlay, (0, 0))

    def draw_minimap(self):
        scale = self.minimap_scale
        map_w = self.map_cols * scale
        map_h = self.map_rows * scale
        surf = self.minimap_base.copy()

        ex, ey = self.exit_cell
        exit_color = (90, 210, 110) if self.exit_unlocked else (220, 80, 80)
        pygame.draw.rect(surf, exit_color, (ex * scale, ey * scale, scale, scale))

        for kx, ky in self.key_cells:
            pygame.draw.rect(surf, (250, 220, 70), (kx * scale, ky * scale, scale, scale))
        for bx, by in self.battery_cells:
            pygame.draw.rect(surf, (80, 220, 255), (bx * scale, by * scale, scale, scale))
        for sx, sy in self.boost_cells:
            pygame.draw.rect(surf, (120, 255, 120), (sx * scale, sy * scale, scale, scale))

        player_x = (self.player_x / TILE_SIZE) * scale
        player_y = (self.player_y / TILE_SIZE) * scale
        monster_x = (self.monster_x / TILE_SIZE) * scale
        monster_y = (self.monster_y / TILE_SIZE) * scale

        # Indicatore direzione: tronco di cono orientato con l'angolo visuale.
        dir_x = math.cos(self.player_angle)
        dir_y = math.sin(self.player_angle)
        perp_x = -dir_y
        perp_y = dir_x
        cone_len = max(16, int(scale * 8.5))
        near_w = 2.0
        far_w = max(6.0, scale * 2.6)

        near_left = (player_x + perp_x * near_w, player_y + perp_y * near_w)
        near_right = (player_x - perp_x * near_w, player_y - perp_y * near_w)
        far_center = (player_x + dir_x * cone_len, player_y + dir_y * cone_len)
        far_left = (far_center[0] + perp_x * far_w, far_center[1] + perp_y * far_w)
        far_right = (far_center[0] - perp_x * far_w, far_center[1] - perp_y * far_w)

        cone_layer = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        pygame.draw.polygon(
            cone_layer,
            (255, 215, 140, 72),
            [(int(near_left[0]), int(near_left[1])), (int(far_left[0]), int(far_left[1])),
             (int(far_right[0]), int(far_right[1])), (int(near_right[0]), int(near_right[1]))],
        )
        pygame.draw.line(
            cone_layer,
            (255, 240, 190, 150),
            (int(player_x), int(player_y)),
            (int(far_center[0]), int(far_center[1])),
            1,
        )
        surf.blit(cone_layer, (0, 0))

        pygame.draw.circle(surf, (95, 190, 255), (int(player_x), int(player_y)), 3)
        pygame.draw.circle(surf, (230, 80, 80), (int(monster_x), int(monster_y)), 3)

        pos = (WIDTH - map_w - 16, 16)
        self.screen.blit(surf, pos)
        pygame.draw.rect(self.screen, (220, 220, 220), (*pos, map_w, map_h), 2)

    def draw_bar(self, x, y, w, h, ratio, fg, bg, label):
        ratio = clamp(ratio, 0.0, 1.0)
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, fg, (x, y, int(w * ratio), h), border_radius=6)
        pygame.draw.rect(self.screen, (10, 10, 10), (x, y, w, h), 2, border_radius=6)
        txt = self.small_font.render(label, True, (240, 240, 240))
        self.screen.blit(txt, (x + 6, y - 18))

    def draw_ui(self):
        self.draw_bar(16, 16, 260, 20, self.stamina / STAMINA_MAX, (80, 220, 110), (30, 55, 38), "Stamina")
        self.draw_bar(16, 50, 260, 20, self.battery / BATTERY_MAX, (80, 200, 255), (30, 40, 55), "Batteria")

        key_text = self.font.render(
            f"Chiavi: {self.keys_collected}/{self.keys_required}",
            True,
            (245, 220, 110),
        )
        timer_text = self.font.render(f"Tempo: {int(self.elapsed)}s", True, (235, 235, 235))
        fps_text = self.small_font.render(f"FPS: {int(self.clock.get_fps())}", True, (210, 210, 210))

        self.screen.blit(key_text, (16, 82))
        self.screen.blit(timer_text, (16, 108))
        self.screen.blit(fps_text, (16, 136))
        if self.monster_grace_timer > 0 and self.state == "playing":
            safe_msg = self.small_font.render(
                f"Mostro inattivo: {self.monster_grace_timer:.1f}s",
                True,
                (255, 205, 130),
            )
            self.screen.blit(safe_msg, (16, 158))

        objective = (
            "Obiettivo: trova le chiavi e poi raggiungi l'uscita"
            if not self.exit_unlocked
            else "Obiettivo: vai all'uscita verde"
        )
        self.screen.blit(self.small_font.render(objective, True, (230, 230, 230)), (16, HEIGHT - 28))

        hint = "WASD muovi | Mouse visuale | SHIFT corsa | F torcia | TAB minimappa | ESC pausa"
        self.screen.blit(self.small_font.render(hint, True, (210, 210, 210)), (16, HEIGHT - 50))

        if self.status_timer > 0 and self.status_text:
            msg = self.font.render(self.status_text, True, (255, 240, 170))
            self.screen.blit(msg, msg.get_rect(center=(WIDTH // 2, 36)))

    def draw_overlay_message(self, title, subtitle, color):
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        layer.fill((0, 0, 0, 170))
        self.screen.blit(layer, (0, 0))

        t1 = self.big_font.render(title, True, color)
        t2 = self.font.render(subtitle, True, (235, 235, 235))
        t3 = self.small_font.render("R = nuova partita | Q = esci", True, (220, 220, 220))

        self.screen.blit(t1, t1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))
        self.screen.blit(t2, t2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 8)))
        self.screen.blit(t3, t3.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 42)))

    def draw_glitch_sprite(self, target, image, rect, t, color_a, color_b):
        target.blit(image, rect)

        # Slice glitch orizzontale.
        w, h = image.get_size()
        for i in range(7):
            y = int((0.5 + 0.5 * math.sin(t * (3.2 + i * 0.6) + i)) * (h - 8))
            sh = 3 + (i % 4)
            if y + sh >= h:
                sh = h - y - 1
            if sh <= 0:
                continue
            strip = image.subsurface((0, y, w, sh)).copy()
            offset = int(math.sin(t * (10.0 + i * 1.5)) * (4 + i))
            target.blit(strip, (rect.x + offset, rect.y + y))

      

    def draw_menu(self):
        t = pygame.time.get_ticks() * 0.001
        self.screen.blit(self.menu_background, (0, 0))
        
        # Colori originali
        blood_red = (255, 20, 30)
        deep_black = (2, 2, 3)
        bright_gold = (255, 200, 80)
        pale_gold = (255, 235, 180)

        fade_alpha = int(255 * clamp((pygame.time.get_ticks() - self.menu_enter_time) / 0.75, 0.0, 1.0))

        ui_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(ui_layer, deep_black, (0, 0, WIDTH, HEIGHT))

        # Scanlines animate - effetto TV glitch
        scanlines = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        scan_offset = int((t * 100) % 4)
        for i in range(0, HEIGHT, 3):
            alpha = 20 + int(10 * math.sin(t * 8 + i * 0.1))
            pygame.draw.line(scanlines, (blood_red[0], blood_red[1], blood_red[2], alpha), (0, i + scan_offset), (WIDTH, i + scan_offset), 1)
        ui_layer.blit(scanlines, (0, 0))

        # Griglia di sfondo - linee
        grid = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for x in range(0, WIDTH, 160):
            alpha = 15 if (x // 160) % 2 == 0 else 8
            pygame.draw.line(grid, (bright_gold[0], bright_gold[1], bright_gold[2], alpha), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 120):
            alpha = 12 if (y // 120) % 2 == 0 else 6
            pygame.draw.line(grid, (blood_red[0], blood_red[1], blood_red[2], alpha), (0, y), (WIDTH, y), 1)
        ui_layer.blit(grid, (0, 0))

        # Titolo principale GRANDE con effetto glow
        title_font = pygame.font.SysFont("consolas", 92, bold=True)
        title_text = "ESCAPE FROM CK"
        
        # Effetto glow multiplo
        for offset, alpha_val in [(4, 100), (8, 60), (12, 30)]:
            title_glow = title_font.render(title_text, True, blood_red)
            title_glow.set_alpha(alpha_val)
            ui_layer.blit(title_glow, (WIDTH // 2 - title_glow.get_width() // 2 + offset - 4, 30 + offset - 4))
        
        title_main = title_font.render(title_text, True, pale_gold)
        title_pos = title_main.get_rect(center=(WIDTH // 2, 60))
        ui_layer.blit(title_main, title_pos)

        # Subtitle grande
        subtitle_font = pygame.font.SysFont("consolas", 18, bold=True)
        subtitle_small = subtitle_font.render("[ SYSTEM ONLINE ] [ THREAT DETECTED ]", True, bright_gold)
        ui_layer.blit(subtitle_small, subtitle_small.get_rect(center=(WIDTH // 2, 110)))

        # Linee separatrici pesanti
        pygame.draw.line(ui_layer, blood_red, (40, 135), (WIDTH - 40, 135), 3)
        pygame.draw.line(ui_layer, bright_gold, (40, 140), (WIDTH - 40, 140), 1)

        # Layout a due colonne: mostro a sinistra, menu a destra
        monster_panel_x = 60
        monster_panel_y = 160
        monster_panel_w = 480
        monster_panel_h = HEIGHT - 280

        menu_panel_x = monster_panel_x + monster_panel_w + 50
        menu_panel_w = WIDTH - menu_panel_x - 60
        menu_panel_y = monster_panel_y
        menu_panel_h = monster_panel_h

        # === PANNELLO MOSTRO CON CORNICE ===
        pygame.draw.rect(ui_layer, blood_red, (monster_panel_x, monster_panel_y, monster_panel_w, monster_panel_h), 3)
        pygame.draw.rect(ui_layer, bright_gold, (monster_panel_x + 2, monster_panel_y + 2, monster_panel_w - 4, monster_panel_h - 4), 1)
        
        # Glitch corners
        corner_size = 20
        for corner_x, corner_y in [(monster_panel_x, monster_panel_y), 
                                   (monster_panel_x + monster_panel_w - corner_size, monster_panel_y),
                                   (monster_panel_x, monster_panel_y + monster_panel_h - corner_size),
                                   (monster_panel_x + monster_panel_w - corner_size, monster_panel_y + monster_panel_h - corner_size)]:
            glitch_color = blood_red if int(t * 5) % 2 == 0 else bright_gold
            pygame.draw.rect(ui_layer, glitch_color, (corner_x, corner_y, corner_size, corner_size), 2)

        # Header mostro GRANDE
        header_font = pygame.font.SysFont("consolas", 18, bold=True)
        header_text = header_font.render(">> TARGET ENTITY <<", True, bright_gold)
        ui_layer.blit(header_text, (monster_panel_x + 20, monster_panel_y + 15))

        # Mostro con effetto pulsante
        monster_size = int(300 + 25 * math.sin(t * 1.8))
        monster_prev = pygame.transform.smoothscale(self.monster_sprite, (monster_size, monster_size))
        monster_center_x = monster_panel_x + monster_panel_w // 2
        monster_center_y = monster_panel_y + monster_panel_h // 2 - 20
        monster_rect = monster_prev.get_rect(center=(monster_center_x, monster_center_y))

        # Glow multiplo intorno al mostro
        for r, col, a in [(80, blood_red, 100), (130, bright_gold, 70), (180, blood_red, 40)]:
            circle_surf = pygame.Surface((r*2+30, r*2+30), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (*col, a), (r+15, r+15), r, 3)
            ui_layer.blit(circle_surf, (monster_center_x - r - 15, monster_center_y - r - 15))

        self.draw_glitch_sprite(ui_layer, monster_prev, monster_rect, t, blood_red, bright_gold)

        # Info mostro GRANDE
        info_font = pygame.font.SysFont("consolas", 16, bold=True)
        info_small = info_font.render("CK", True, blood_red)
        ui_layer.blit(info_small, (monster_panel_x + 20, monster_panel_y + monster_panel_h - 50))
        
        threat_font = pygame.font.SysFont("consolas", 15)
        threat_small = threat_font.render("STATUS: ARMED", True, bright_gold)
        ui_layer.blit(threat_small, (monster_panel_x + 20, monster_panel_y + monster_panel_h - 25))

        # === PANNELLO MENU A DESTRA ===
        pygame.draw.rect(ui_layer, bright_gold, (menu_panel_x, menu_panel_y, menu_panel_w, menu_panel_h), 3)
        pygame.draw.rect(ui_layer, blood_red, (menu_panel_x + 2, menu_panel_y + 2, menu_panel_w - 4, menu_panel_h - 4), 1)

        # Header menu GRANDE
        menu_header = header_font.render(">> CONTROL SYSTEM <<", True, blood_red)
        ui_layer.blit(menu_header, (menu_panel_x + 20, menu_panel_y + 15))

        options = [
            ("START GAME", ""),
            ("DIFFICULTY", self.current_difficulty_name()),
            ("MUSIC VOLUME", f"{self.menu_music_volume}%"),
            ("SFX VOLUME", f"{self.menu_sfx_volume}%"),
            ("MINIMAP", "ON" if self.menu_minimap_default else "OFF"),
            ("FULLSCREEN", "ON" if self.fullscreen else "OFF"),
            ("EXIT", ""),
        ]

        menu_start_y = menu_panel_y + 60
        item_h = (menu_panel_h - 70) // len(options)

        for idx, (label, value) in enumerate(options):
            selected = idx == self.menu_index
            item_y = menu_start_y + idx * item_h
            item_rect = pygame.Rect(menu_panel_x + 18, item_y, menu_panel_w - 36, item_h - 10)

            # Background item
            if selected:
                item_bg = pygame.Surface((item_rect.w + 12, item_rect.h + 12), pygame.SRCALPHA)
                pygame.draw.rect(item_bg, (blood_red[0], blood_red[1], blood_red[2], 60), (6, 6, item_rect.w, item_rect.h), border_radius=6)
                ui_layer.blit(item_bg, (item_rect.x - 6, item_rect.y - 6))
                
                # Border glow
                pygame.draw.rect(ui_layer, bright_gold, item_rect, 3, border_radius=6)
                pygame.draw.rect(ui_layer, blood_red, (item_rect.x + 1, item_rect.y + 1, item_rect.w - 2, item_rect.h - 2), 1, border_radius=6)
                
                # Glitch effect
                if int(t * 7) % 2 == 0:
                    pygame.draw.rect(ui_layer, (255, 100, 110), item_rect, 1, border_radius=6)
            else:
                pygame.draw.rect(ui_layer, (20, 20, 25), item_rect, border_radius=6)
                pygame.draw.rect(ui_layer, bright_gold, item_rect, 1, border_radius=6)

            # Text label GRANDE
            label_color = pale_gold if selected else bright_gold
            label_font_item = pygame.font.SysFont("consolas", 17, bold=selected)
            label_surf = label_font_item.render(label, True, label_color)
            ui_layer.blit(label_surf, (item_rect.x + 15, item_rect.y + item_rect.h // 2 - 11))

            # Value GRANDE
            if value:
                val_color = blood_red if selected else bright_gold
                val_font_item = pygame.font.SysFont("consolas", 16, bold=selected)
                val_surf = val_font_item.render(f"[ {value} ]", True, val_color)
                val_rect = val_surf.get_rect(right=item_rect.right - 15, centery=item_rect.centery)
                ui_layer.blit(val_surf, val_rect)

        # Footer pesante
        pygame.draw.line(ui_layer, blood_red, (40, HEIGHT - 80), (WIDTH - 40, HEIGHT - 80), 2)
        
        footer_font = pygame.font.SysFont("consolas", 14, bold=True)
        footer_text = footer_font.render("NAVIGATION: [ARROW KEYS / WASD]  |  SELECT: [ENTER]  |  QUIT: [ESC]  |  FULLSCREEN: [F11]", True, pale_gold)
        footer_pos = footer_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        ui_layer.blit(footer_text, footer_pos)

        # Audio warning GRANDE
        if self.missing_audio:
            warn_text = "! " + " ".join(self.missing_audio) + " !"
            warn_font = pygame.font.SysFont("consolas", 14, bold=True)
            warn = warn_font.render(warn_text, True, blood_red)
            ui_layer.blit(warn, warn.get_rect(center=(WIDTH // 2, HEIGHT - 20)))

        if fade_alpha < 255:
            ui_layer.set_alpha(fade_alpha)
        self.screen.blit(ui_layer, (0, 0))

    def update(self, dt):
        if self.scene == "menu":
            return
        if self.state != "playing" or self.paused:
            return

        self.elapsed += dt
        self.status_timer = max(0.0, self.status_timer - dt)

        self.update_player(dt)
        self.collect_pickups()
        self.update_monster(dt)
        self.check_goal()

    def draw(self):
        if self.scene == "menu":
            self.draw_menu()
            return

        self.raycast_walls()
        self.draw_sprites()
        self.draw_flashlight()
        if self.show_minimap:
            self.draw_minimap()
        self.draw_ui()

        if self.paused and self.state == "playing":
            self.draw_overlay_message("PAUSA", "ESC per continuare", (255, 255, 255))
        elif self.state == "win":
            self.draw_overlay_message("SEI SOPRAVVISSUTO", "Hai trovato l'uscita", (100, 255, 120))
        elif self.state == "lose":
            self.draw_overlay_message("SEI STATO PRESO", "Il mostro ti ha raggiunto", (255, 90, 90))

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)
            self.handle_events()
            self.update(dt)
            self.update_audio_state()
            self.draw()
            self.present_frame()

        if self.audio_enabled:
            self.set_music_track(None)
            self.stop_heartbeat()
        pygame.quit()


if __name__ == "__main__":
    Escape3DGame().run()
