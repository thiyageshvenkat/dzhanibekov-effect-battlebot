import argparse
import math
import os
import random
import struct
import time
from dataclasses import dataclass

os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)

import pybullet as p

Vec3 = list[float]
Triangle = tuple[Vec3, Vec3, Vec3]

STL_FILE = "assembly.stl"
RAM_STL_FILE = "front_flipper.stl"
SCALE = [0.001, 0.001, 0.001]

RUN_WITH_GUI = True
RECORD_VIDEO = False
VIDEO_FILE = "flip_recording.mp4"

FAKE_MASS_KG = 1.36
FAKE_INERTIA = [0.002, 0.006, 0.0075]

DT = 1 / 1000
SUBSTEPS = 4
FRAMES = 375
TRIALS = 20
REALTIME_PLAYBACK = True
PAUSE_BETWEEN_TRIALS = 0.4

ARENA_HALF_SIZE = 5.0
GROUND_CLEARANCE = 0.015

RAM_START_FRAME = 30
RAM_MASS = 0.7
RAM_HALF_EXTENTS = [0.08, 0.12, 0.04]
RAM_START_GAP = 0.04
RAM_SPEED = 16.0
RAM_SPEED_RANDOMNESS = 0.06
RAM_UP_SPEED = 5.0
RAM_PITCH_UP_SPEED = 55.0
RAM_Y_RANDOMNESS = 0.015
RAM_Z_OFFSET = -0.04
RAM_RESTITUTION = 0.12

PRE_SPIN_Y = 80.0
PRE_SPIN_Y_RANDOMNESS = 3.0
SPIN_APPLY_MIN_HEIGHT = 0.03

MARKER_SIZE = 0.07
AXIS_LINE_LENGTH = 0.65
CAMERA_DISTANCE = 2.7
CAMERA_YAW = 42
CAMERA_PITCH = -24
CAMERA_EVERY = 3
TRAILS_ENABLED = True
TRAIL_EVERY = 2
TRAIL_LIFE = 1.8
TRAIL_WIDTH = 5
SNAP_ENABLED = True
SNAP_LIFE = 0.8
SNAP_MIN_OMEGA_Y = 8.0
AXIS_EVERY = 4
AXIS_LIFE = 0.12
PRINT_EVERY = 60


@dataclass
class Bounds:
    mins: Vec3
    center: Vec3
    half_extents: Vec3


@dataclass
class MassProps:
    source: str
    mass: float
    center_of_mass: Vec3
    inertia: Vec3
    orientation: list[float]


@dataclass
class Scene:
    bot: int
    ram: int
    floor: int
    front_marker: int
    top_marker: int
    side_marker: int
    start: Vec3
    half_extents: Vec3
    collision_offset: Vec3


@dataclass
class Trial:
    ram_position: Vec3
    ram_velocity: Vec3
    ram_angular_velocity: Vec3
    spin: float


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return [a[i] + b[i] for i in range(3)]


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return [a[i] - b[i] for i in range(3)]


def vmul(v: Vec3, s: float) -> Vec3:
    return [x * s for x in v]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(a[i] * b[i] for i in range(3))


def local_to_world(pos: Vec3, axes: tuple[Vec3, Vec3, Vec3], local: Vec3) -> Vec3:
    return vadd(
        pos,
        vadd(vadd(vmul(axes[0], local[0]), vmul(axes[1], local[1])), vmul(axes[2], local[2])),
    )


def body_axes(orientation: list[float]) -> tuple[Vec3, Vec3, Vec3]:
    r = p.getMatrixFromQuaternion(orientation)
    return [r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]


def read_stl(path: str, scale: Vec3) -> list[Triangle]:
    with open(path, "rb") as f:
        f.read(80)
        count_bytes = f.read(4)
        if len(count_bytes) == 4:
            count = struct.unpack("<I", count_bytes)[0]
            if os.path.getsize(path) == 84 + count * 50:
                tris = []
                for _ in range(count):
                    vals = struct.unpack("<12fH", f.read(50))
                    tris.append(
                        tuple(
                            [vals[i] * scale[0], vals[i + 1] * scale[1], vals[i + 2] * scale[2]]
                            for i in (3, 6, 9)
                        )
                    )
                return tris

    tris = []
    current = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                current.append([float(parts[i + 1]) * scale[i] for i in range(3)])
                if len(current) == 3:
                    tris.append(tuple(current))
                    current = []
    return tris


def get_bounds(tris: list[Triangle]) -> Bounds:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for tri in tris:
        for vertex in tri:
            for i in range(3):
                mins[i] = min(mins[i], vertex[i])
                maxs[i] = max(maxs[i], vertex[i])

    return Bounds(
        mins=mins,
        center=[(mins[i] + maxs[i]) / 2 for i in range(3)],
        half_extents=[(maxs[i] - mins[i]) / 2 for i in range(3)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STL flipper collision simulation.")
    parser.add_argument("--record", action="store_true", help="Record the PyBullet GUI to an MP4.")
    parser.add_argument("--video-file", default=VIDEO_FILE, help=f"MP4 output path. Default: {VIDEO_FILE}")
    return parser.parse_args()


def make_body(
    mass: float,
    collision: int,
    visual: int,
    position: Vec3,
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> int:
    return p.createMultiBody(mass, collision, visual, position, orientation)


def make_marker(radius: float, color: list[float]) -> int:
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return make_body(0, -1, visual, [0, 0, 0])


def setup_world(record: bool, video_file: str, start_z: float) -> int:
    p.connect(p.GUI if RUN_WITH_GUI else p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    if not RUN_WITH_GUI:
        return -1

    p.resetDebugVisualizerCamera(2.6, 40, -25, [0, 0, start_z / 2])
    if not record:
        return -1

    video_path = os.path.abspath(video_file)
    print(f"Recording PyBullet GUI video to {video_path}")
    return p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)


def make_scene(mass_props: MassProps, bounds: Bounds) -> Scene:
    center_of_mass_offset = vmul(mass_props.center_of_mass, -1)
    start_z = mass_props.center_of_mass[2] - bounds.mins[2] + GROUND_CLEARANCE
    start = [0.0, 0.0, start_z]

    floor_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ARENA_HALF_SIZE, ARENA_HALF_SIZE, 0.05])
    floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[ARENA_HALF_SIZE, ARENA_HALF_SIZE, 0.05], rgbaColor=[0.18, 0.18, 0.18, 1])
    floor = make_body(0, floor_collision, floor_visual, [0, 0, -0.05])
    p.changeDynamics(floor, -1, lateralFriction=0.9, restitution=0.25)

    bot_visual = p.createVisualShape(
        p.GEOM_MESH,
        fileName=STL_FILE,
        meshScale=SCALE,
        rgbaColor=[0.65, 0.65, 0.65, 1],
        visualFramePosition=center_of_mass_offset,
        visualFrameOrientation=mass_props.orientation,
    )
    bot_collision = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=STL_FILE,
        meshScale=SCALE,
        collisionFramePosition=center_of_mass_offset,
        collisionFrameOrientation=mass_props.orientation,
    )
    bot = make_body(mass_props.mass, bot_collision, bot_visual, start)
    p.changeDynamics(
        bot,
        -1,
        localInertiaDiagonal=mass_props.inertia,
        linearDamping=0,
        angularDamping=0,
        restitution=0.25,
        lateralFriction=0.9,
        rollingFriction=0.02,
        spinningFriction=0.02,
    )

    ram_collision = p.createCollisionShape(p.GEOM_MESH, fileName=RAM_STL_FILE, meshScale=[1.0, 1.0, 1.0])
    ram_visual = p.createVisualShape(p.GEOM_MESH, fileName=RAM_STL_FILE, meshScale=[1.0, 1.0, 1.0], rgbaColor=[1, 0.85, 0.05, 1])
    ram = make_body(RAM_MASS, ram_collision, ram_visual, [0, 4, RAM_HALF_EXTENTS[2]])
    p.changeDynamics(ram, -1, linearDamping=0, angularDamping=0, lateralFriction=0.8, restitution=RAM_RESTITUTION)

    half_extents = [max(x, 0.015) for x in bounds.half_extents]
    collision_offset = vsub(bounds.center, mass_props.center_of_mass)

    return Scene(
        bot=bot,
        ram=ram,
        floor=floor,
        front_marker=make_marker(MARKER_SIZE, [1, 0, 0, 1]),
        top_marker=make_marker(MARKER_SIZE * 0.85, [0, 0, 1, 1]),
        side_marker=make_marker(MARKER_SIZE * 0.75, [0, 1, 0, 1]),
        start=start,
        half_extents=half_extents,
        collision_offset=collision_offset,
    )


def print_summary(mass_props: MassProps) -> None:
    ordered = sorted(zip(mass_props.inertia, "XYZ"))
    print("Running full launch-and-landing simulation.")
    print(f"Mass property source: {mass_props.source}")
    print(f"MASS = {mass_props.mass}")
    print(f"Center of mass = {mass_props.center_of_mass}")
    print(f"INERTIA = {mass_props.inertia}")
    print(f"Axis order: {ordered[0][1]} < {ordered[1][1]} < {ordered[2][1]}")
    print(f"Intermediate axis is {ordered[1][1]}.")
    print(f"One-axis spin Y = {PRE_SPIN_Y} +/- {PRE_SPIN_Y_RANDOMNESS}")
    print(f"Ram mass = {RAM_MASS}")
    print(f"Ram speed = {RAM_SPEED} +/- {RAM_SPEED * RAM_SPEED_RANDOMNESS}")
    print(f"Ram launch frame = {RAM_START_FRAME}")
    print(f"Real-time playback = {REALTIME_PLAYBACK}; {SUBSTEPS * DT:.4f}s simulated per visible frame.")
    print("Watch red/front and blue/top markers for snap/reversal.")


def reset_trial(scene: Scene) -> Trial:
    p.removeAllUserDebugItems()
    p.resetBasePositionAndOrientation(scene.bot, scene.start, [0, 0, 0, 1])
    p.resetBaseVelocity(scene.bot, [0, 0, 0], [0, 0, 0])

    ram_speed = RAM_SPEED * random.uniform(1 - RAM_SPEED_RANDOMNESS, 1 + RAM_SPEED_RANDOMNESS)
    ram_y = random.uniform(-RAM_Y_RANDOMNESS, RAM_Y_RANDOMNESS)
    bot_front_x = scene.start[0] + scene.collision_offset[0] + scene.half_extents[0]
    ram_position = [
        bot_front_x + RAM_HALF_EXTENTS[0] + RAM_START_GAP,
        scene.start[1] + scene.collision_offset[1] + ram_y,
        max(RAM_HALF_EXTENTS[2], scene.start[2] + scene.collision_offset[2] + RAM_Z_OFFSET),
    ]
    trial = Trial(
        ram_position=ram_position,
        ram_velocity=[-ram_speed, 0.0, RAM_UP_SPEED],
        ram_angular_velocity=[0.0, RAM_PITCH_UP_SPEED, 0.0],
        spin=random.uniform(PRE_SPIN_Y - PRE_SPIN_Y_RANDOMNESS, PRE_SPIN_Y + PRE_SPIN_Y_RANDOMNESS),
    )

    p.resetBasePositionAndOrientation(scene.ram, trial.ram_position, [0, 0, 0, 1])
    p.resetBaseVelocity(scene.ram, [0, 0, 0], [0, 0, 0])

    print(f"ram start = {trial.ram_position}")
    print(f"ram velocity at launch = {trial.ram_velocity}")
    print(f"ram angular velocity at launch = {trial.ram_angular_velocity}")
    print(f"one-axis spin once airborne = {[0, trial.spin, 0]}")
    return trial


def draw_trails(marker_positions: list[Vec3], previous: list[Vec3]) -> None:
    if not (RUN_WITH_GUI and TRAILS_ENABLED and previous):
        return
    for old, new, color in zip(previous, marker_positions, ([1, 0, 0], [0, 0, 1], [0, 1, 0])):
        p.addUserDebugLine(old, new, color, TRAIL_WIDTH, TRAIL_LIFE)


def draw_axes(pos: Vec3, axes: tuple[Vec3, Vec3, Vec3]) -> None:
    if not RUN_WITH_GUI:
        return
    for axis, color in zip(axes, ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        p.addUserDebugLine(pos, vadd(pos, vmul(axis, AXIS_LINE_LENGTH)), color, 3, AXIS_LIFE)


def run_trial(scene: Scene, trial_index: int) -> None:
    print(f"\n=== TRIAL {trial_index + 1}/{TRIALS} ===")
    trial = reset_trial(scene)
    previous_markers: list[Vec3] = []
    previous_omega_y = 0.0
    have_previous_omega = False
    spin_applied = False
    wall_start = time.perf_counter()

    for frame in range(FRAMES):
        for _ in range(SUBSTEPS):
            if frame == RAM_START_FRAME:
                p.resetBaseVelocity(scene.ram, trial.ram_velocity, trial.ram_angular_velocity)
                if RUN_WITH_GUI:
                    p.addUserDebugLine(trial.ram_position, vadd(trial.ram_position, vmul(trial.ram_velocity, 0.035)), [1, 1, 0], 6, 0.4)
            p.stepSimulation()

        pos, orientation = p.getBasePositionAndOrientation(scene.bot)
        linear_velocity, angular_velocity = p.getBaseVelocity(scene.bot)

        if not spin_applied:
            contacts = p.getContactPoints(scene.bot, scene.floor) + p.getContactPoints(scene.bot, scene.ram)
            if pos[2] > scene.start[2] + SPIN_APPLY_MIN_HEIGHT and not contacts:
                y_axis = body_axes(orientation)[1]
                p.resetBaseVelocity(scene.bot, linear_velocity, vmul(y_axis, trial.spin))
                _, angular_velocity = p.getBaseVelocity(scene.bot)
                spin_applied = True
                print(f">>> AIRBORNE ONE-AXIS SPIN APPLIED at frame {frame}")

        axes = body_axes(orientation)
        omega = [dot(angular_velocity, axis) for axis in axes]
        marker_positions = [
            local_to_world(pos, axes, [0.55, 0, 0]),
            local_to_world(pos, axes, [0, 0, 0.25]),
            local_to_world(pos, axes, [0, 0.32, 0]),
        ]

        if RUN_WITH_GUI and frame % CAMERA_EVERY == 0:
            p.resetDebugVisualizerCamera(CAMERA_DISTANCE, CAMERA_YAW, CAMERA_PITCH, pos)
        if frame % TRAIL_EVERY == 0:
            draw_trails(marker_positions, previous_markers)
            previous_markers = marker_positions
        if frame % AXIS_EVERY == 0:
            draw_axes(pos, axes)

        if SNAP_ENABLED and have_previous_omega:
            snap = previous_omega_y * omega[1] < 0 and abs(previous_omega_y) > SNAP_MIN_OMEGA_Y and abs(omega[1]) > SNAP_MIN_OMEGA_Y
            if pos[2] > scene.start[2] + 0.12 and snap:
                print(">>> SNAP FLIP")
                p.addUserDebugText("SNAP FLIP", pos, textColorRGB=[1, 1, 0], textSize=2.4, lifeTime=SNAP_LIFE)
        previous_omega_y = omega[1]
        have_previous_omega = True

        for body_id, marker_pos in zip((scene.front_marker, scene.top_marker, scene.side_marker), marker_positions):
            p.resetBasePositionAndOrientation(body_id, marker_pos, [0, 0, 0, 1])

        if frame % PRINT_EVERY == 0:
            print(f"frame {frame:4d}: z={pos[2]:7.3f}  body omega = ({omega[0]:7.3f}, {omega[1]:7.3f}, {omega[2]:7.3f})")

        if RUN_WITH_GUI and REALTIME_PLAYBACK:
            remaining = wall_start + (frame + 1) * SUBSTEPS * DT - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    tris = read_stl(STL_FILE, SCALE)
    bounds = get_bounds(tris)

    mass_props = MassProps("fake theorem demo", FAKE_MASS_KG, [0.0, 0.0, 0.0], FAKE_INERTIA, [0.0, 0.0, 0.0, 1.0])
    start_z = (mass_props.center_of_mass[2] - bounds.mins[2]) + GROUND_CLEARANCE
    video_log_id = setup_world(args.record or RECORD_VIDEO, args.video_file, start_z)
    scene = make_scene(mass_props, bounds)
    print_summary(mass_props)

    for trial_index in range(TRIALS):
        run_trial(scene, trial_index)
        time.sleep(PAUSE_BETWEEN_TRIALS)

    if video_log_id >= 0:
        p.stopStateLogging(video_log_id)
        print(f"Saved PyBullet GUI video to {os.path.abspath(args.video_file)}")
    p.disconnect()


if __name__ == "__main__":
    main()
