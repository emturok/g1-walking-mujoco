import sys
import mujoco, mujoco.viewer

xml_path = sys.argv[1]  # the path you type after the script name
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

print("nq", m.nq, "nv", m.nv, "nu", m.nu, "dt", m.opt.timestep)
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
