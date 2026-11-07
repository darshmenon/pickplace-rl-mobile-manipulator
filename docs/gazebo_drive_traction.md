# Driving the base in Gazebo without breaking the grasp

Under `bullet-featherstone` (Gazebo Harmonic) the diff-drive base and the Robotiq grasp could not be had together with wheel traction. Every claim below was checked against Gazebo's real poses (`/world/pickplace_world/pose/info`), not against odometry or the controller log, because odometry kept reporting progress while the base stood still.

| Wheels | Base drives | Grasp and carry |
|--------|-------------|-----------------|
| 0.5 kg, friction 1 (stock) | No | Yes |
| 0.5 kg, friction 300 | No (odom said 0.771, real base stayed at x = -0.8) | Yes |
| 5 kg | No | not reached |
| 15 kg, friction 1 | Yes | No (fingers never close firmly on the cube) |
| 15 kg, casters at the original height | Yes | No (caster height is not the cause) |

Friction does not change the result, so the high wheel friction used by the MiR robot in other repos does not carry over to Featherstone. The grasp failure with heavy wheels has no confirmed cause.

## What works

`drive:=true` uses `mobile_ur3_vel.urdf`: the diff-drive plugin is replaced by Gazebo's velocity-control plugin plus an odometry publisher, so `/cmd_vel` pushes the model directly and the light wheels stay. `drive_to` then creeps to the pick spot.

A second bug showed up here: stopping 8 mm short of the pick spot (1 cm tolerance) left the pinch more than 4 mm from the cube, so the approach never converged. A 2 mm tolerance with a 2 cm/s creep fixed it. In the checked run the base stopped at x = -0.0023 and the real cube was carried to the target at (0.488, 0.138).

`drive_mode:=glide` moves the base with `set_pose` instead and worked in 1 of 2 runs.

## Not tried

DART drives the base but ignores `<mimic>` joints, so the finger gap collapses. Commanding the second finger explicitly would let DART be used with real wheels.
