"""dronelab – a CPU-only quadrotor autonomy stack, one layer per subproject.

core        simulator backend, courses, disturbances, evaluation (shared by all subprojects)
control     trajectory-tracking controllers and the tuning protocol
surrogate   learned dynamics models
autonomy    gate flight without a given reference path
estimation  state estimation from IMU and position measurements
vision      gate detection from onboard camera images
"""

__version__ = "0.2.0"
