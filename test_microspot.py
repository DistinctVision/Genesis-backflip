import typing as tp
from pathlib import Path
from pynput import keyboard
from threading import RLock
import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from rsl_rl.env import VecEnv

pressed_locker = RLock()
pressed: tp.Set[str] = set()


def on_press(key):
    global pressed
    
    if hasattr(key, "char"):
        key_str = str(key.char)
    else:
        key_str = str(key)
    with pressed_locker:
        pressed.add(key_str)


def on_release(key):
    global pressed
    
    if hasattr(key, "char"):
        key_str = str(key.char)
    else:
        key_str = str(key)
    with pressed_locker:
        pressed.remove(key_str)


def run_microspot_scene(dt: float = 1e-2):
    device = torch.device("cuda")

    dof_names = ['front_left_shoulder', 'front_right_shoulder',
                 'rear_left_shoulder', 'rear_right_shoulder',
                 'front_left_leg', 'front_right_leg',
                 'rear_left_leg', 'rear_right_leg',
                 'front_left_foot', 'front_right_foot',
                 'rear_left_foot', 'rear_right_foot']

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(1 / dt),
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            n_rendered_envs=1,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    rigid_solver: RigidSolver | None = None
    for solver in scene.sim.solvers:
        if not isinstance(solver, RigidSolver):
            continue
        rigid_solver = solver

    scene.add_entity(
        gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True),
    )

    floating_camera = scene.add_camera(
        pos=np.array([0, -1, 1]),
        lookat=np.array([0, 0, 0]),
        # res=(720, 480),
        fov=40,
        GUI=True,
    )

    base_init_positon = np.array([0, 0, 1], dtype=np.float32)
    base_init_q = np.array([1, 0, 0, 0], dtype=np.float32)

    robot: RigidEntity = scene.add_entity(
        gs.morphs.URDF(
            file=str(Path("urdf") / "spotmicro.urdf"),
            merge_fixed_links=False,
            links_to_keep=[],
            pos=base_init_positon,
            quat=base_init_q,
        ),
        visualize_contact=False,
    )

    motor_dofs = []
    dof_limits = []
    for dof_name in dof_names:
        joint: RigidJoint = robot.get_joint(dof_name)
        print(joint.dofs_limit)
        motor_dofs.append(joint.dof_idx_local)
        dof_limits.append(joint.dofs_limit[0])
    dof_limits = torch.tensor(dof_limits, device=device, dtype=gs.tc_float)
    print(f"{dof_limits=}")

    speed = 1e-2
    kb_actions = {
        "1": (0, speed),
        "2": (0, -speed),
        "3": (1, speed),
        "4": (1, -speed),
        "5": (2, speed),
        "6": (2, -speed),
        "7": (3, speed),
        "8": (3, -speed),
        "q": (4, speed),
        "w": (4, -speed),
        "e": (5, speed),
        "r": (5, -speed),
        "t": (6, speed),
        "y": (6, -speed),
        "u": (7, speed),
        "i": (7, -speed),
        "a": (8, speed),
        "s": (8, -speed),
        "d": (9, speed),
        "f": (9, -speed),
        "g": (10, speed),
        "h": (10, -speed),
        "j": (11, speed),
        "k": (11, -speed),
    }
    
    scene.build(n_envs=1)

    target_dof_pos = torch.zeros(1, len(motor_dofs), device=device, dtype=gs.tc_float)

    while True:
        
        with pressed_locker:
            for pressed_str in pressed:
                if pressed_str in kb_actions:
                    dof_idx, speed = kb_actions[pressed_str]
                    target_dof_pos[:, dof_idx] += speed

        robot.control_dofs_position(position=target_dof_pos, dofs_idx_local=motor_dofs)
        scene.step()


if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level='info')

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    run_microspot_scene()
