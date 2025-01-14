"""This file contains the Parameter class and the Model class."""

class Parameter:
    def __init__(self, value: float, min: float, max: float, optimize: bool = True):
        """Initialize the parameter.

        Args:
            value: The current value of the parameter.
            min: The minimum value of the parameter.
            max: The maximum value of the parameter.
            optimize: Whether the parameter should be optimized.
        """
        self.value: float = value
        self.min: float = min
        self.max: float = max

        self.optimize: bool = optimize


class Model:
    def __init__(self):
        """Initialize the model."""
        self.stiffness = Parameter(14.0, 12.0, 32.0)
        self.damping = Parameter(1.95, 0.3, 6.0)
        self.armature = Parameter(0.005, 0.001, 0.01)
    
        self.friction_static = Parameter(0.01, 0.0011, 0.02)
        self.friction_dynamic = Parameter(0.01, 0.0011, 0.02)
        # TODO
        self.velocity = Parameter(2.0, 0.9, 6.0)

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
