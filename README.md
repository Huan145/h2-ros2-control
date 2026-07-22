# h2-ros2-control

ROS 2 control stack for the Unitree H2 humanoid in Isaac Sim.
One ROS 2 system; nodes can be distributed across machines via the bridge config.

## Status
Documentation in progress.

## Layout
- `nodes/` — ROS 2 nodes (control, perception, custom bridge)
- `launch/` — launch files
- `config/` — configuration, incl. ros_bridge_config.yaml
- `docker/` — perception container (FoundationPose)
- `docs/` — deeper documentation
