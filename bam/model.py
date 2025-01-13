"""This file contains the Parameter class and the Model class."""


class Parameter:
    def __init__(self, value: float, min: float, max: float, optimize: bool = True):
        # Current value of the parameter
        self.value: float = value

        # Minimum and maximum values for the parameter
        self.min: float = min
        self.max: float = max

        # Should this parameter be optimized?
        self.optimize: bool = optimize


class Model:
    def __init__(self):
        self.stiffness = Parameter(15.0, 10.0, 25.0)
        self.damping = Parameter(1.0, 0.3, 2.0)
        self.armature = Parameter(0.005, 0.001, 2.0)
        self.friction = Parameter(0.1, 0.0011, 0.3)

    def initialize(self):
        # Torque constant [Nm/A] or [V/(rad/s)]
        self.model.damping = Parameter(1.6, 1.0, 15.0)

        # Motor resistance [Ohm]
        self.model.stiffness = Parameter(2.0, 0.1, 3.5)

        self.model.friction = Parameter(0.1, 0.0, 0.01)

        # Motor armature / apparent inertia [kg m^2]
        self.model.armature = Parameter(0.005, 0.001, 2.0)

    def get_parameters(self) -> dict:
        """
        This returns the list of parameters that can be optimized.
        """
        return {
            name: param
            for name, param in vars(self).items()
            if isinstance(param, Parameter)
        }

    def get_parameter_values(self) -> dict:
        """
        Return a dict containing parameter values
        """
        parameters = self.get_parameters()
        x = {}
        for name in parameters:
            parameter = parameters[name]
            if parameter.optimize:
                x[name] = parameter.value
        return x


if __name__ == "__main__":
    model = Model()
    print(model.get_parameters())
