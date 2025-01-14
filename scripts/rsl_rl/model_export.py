"""Script to export the model to kinfer format."""
from kinfer import proto as P

ACTION_NUMBER = 10

JOINT_NAMES = [
    "L_Hip_Pitch", 
    "L_Hip_Roll", 
    "L_Hip_Yaw", 
    "L_Knee_Pitch", 
    "L_Ankle_Pitch", 
    "R_Hip_Pitch", 
    "R_Hip_Roll", 
    "R_Hip_Yaw", 
    "R_Knee_Pitch", 
    "R_Ankle_Pitch"

]

input_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="velocity_command",
            vector_command=P.VectorCommandSchema(
                dimensions=3,  # x_vel, y_vel, rot
            ),
        ),
        # Abusing the IMU schema to pass projected gravity instead of raw sensor data
        P.ValueSchema(
            value_name="projected_gravity",
            imu=P.ImuSchema(
                use_accelerometer=True,
                use_gyroscope=False,
                use_magnetometer=False,
            ),
        ),
        P.ValueSchema(
            value_name="joint_angles",
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointPositionUnit.RADIANS,
            ),
        ),
        P.ValueSchema(
            value_name="joint_velocities",
            joint_velocities=P.JointVelocitiesSchema(
                joint_names=JOINT_NAMES,
                unit=P.JointVelocityUnit.RADIANS_PER_SECOND,
            ),
        ),
        P.ValueSchema(
            value_name="actions",
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES, unit=P.JointPositionUnit.RADIANS
            ),
        ),
    ]
)


input_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="input",
            state_tensor=P.StateTensorSchema(
                shape=[36],
                dtype=P.DType.FP32,
            ),
        ),
    ]
)

output_schema = P.IOSchema(
    values=[
        P.ValueSchema(
            value_name="actions",
            joint_positions=P.JointPositionsSchema(
                joint_names=JOINT_NAMES, unit=P.JointPositionUnit.RADIANS
            ),
        ),
    ]
)