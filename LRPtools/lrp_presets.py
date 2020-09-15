import dataclasses
from dataclasses import dataclass


@dataclass
class LRPPreset(object):
    """
    lrp_method LRP method to use in all but the "special" layers
    lrp_method_input LRP method to use in the input layer
    lrp_method_linear LRP method to use in the Linear layers
    """
    lrp_method: str = "epsilon"
    lrp_method_input: str = "epsilon"
    lrp_method_linear: str = "epsilon"
    lrp_method_batchnorm: str ="alphabetax"# "alphabetax"
    lrp_method_relu: str = 'identity'
    lrp_params: dict = dataclasses.field(default_factory=dict)


@dataclass
class IgnoreBias(object):
    # TODO Multi inheritance
    def __post_init__(self):
        self.lrp_params["ignore_bias"] = True


@dataclass
class Epsilon(LRPPreset):
    lrp_method: str = "epsilon"
    lrp_method_input: str = "epsilon"
    lrp_method_linear: str = "epsilon"
    lrp_method_batchnorm: str = "alphabetax"

    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["ignore_bias"] = False


@dataclass
class EpsilonIB(Epsilon):
    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["ignore_bias"] = True


@dataclass
class SequentialPreset(LRPPreset):
    lrp_method: str = "alpha_beta"
    lrp_method_input: str = "alpha_beta"

    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 1
        self.lrp_params["beta"] = 0


@dataclass
class Alpha1Beta0IB(LRPPreset):
    lrp_method: str = "alpha_beta"
    lrp_method_input: str = "alpha_beta"
    lrp_method_linear: str = "alpha_beta"

    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 1
        self.lrp_params["beta"] = 0
        self.lrp_params["ignore_bias"] = True


@dataclass
class Alpha2Beta1IB(LRPPreset):
    lrp_method: str = "alpha_beta"
    lrp_method_input: str = "alpha_beta"
    lrp_method_linear: str = "alpha_beta"

    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 2
        self.lrp_params["beta"] = 1
        self.lrp_params["ignore_bias"] = True


@dataclass
class DTD(LRPPreset):
    lrp_method: str = "z+"
    lrp_method_input: str = "z+"
    lrp_method_linear: str = "z+"

    def __post_init__(self):
        self.lrp_params["ignore_bias"] = True


@dataclass
class DTDBounded(DTD):
    lrp_method_input: str = "zB"


@dataclass
class SequentialPresetA(LRPPreset):
    lrp_method: str = "alpha_beta"
    lrp_method_input: str = "alpha_beta"
    lrp_method_linear: str = "epsilon"
    lrp_method_relu: str = 'alpha_beta'
    lrp_method_batchnorm: str = "identity"
    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 1
        self.lrp_params["beta"] = 0
        self.lrp_params["ignore_bias"] = True

@dataclass
class SequentialPresetB(SequentialPreset):
    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 2
        self.lrp_params["beta"] = 1


def get_preset(lrp_config, preset_kwargs=None):
    """Map string to preset with optional keyword arguments"""
    if preset_kwargs is None:
        preset_kwargs = dict()
    return {
        "lrp.sequential_preset_a": SequentialPresetA,
        "lrp.sequential_preset_b": SequentialPresetB,
        "lrp.alpha_1_beta_0_IB": Alpha1Beta0IB,
        "lrp.alpha_2_beta_1_IB": Alpha2Beta1IB,
        "lrp.epsilon": Epsilon,
        "lrp.epsilon_IB": EpsilonIB,
        "deep_taylor": DTD,
        "deep_taylor.bounded": DTDBounded,
    }[lrp_config](**preset_kwargs)


def innvestigate_presets(lrp_config):
    # iNNvestigate names
    lrp_method, lrp_method_input, lrp_method_linear, lrp_method_batchnorm, lrp_params = \
        {
            "lrp.sequential_preset_a":  # Z+ instead of alpha1beta0
                ("z+", "zB", "epsilon", "alphabetax", {}),
            "lrp.sequential_preset_a2":
                ("alpha_beta", "alpha_beta", "epsilon", "alphabetax",
                 {"alpha": 1, "beta": 0}),
            "lrp.sequential_preset_b":
                ("alpha_beta", "alpha_beta", "epsilon", "alphabetax",
                 {"alpha": 2, "beta": 1}),
            "lrp.alpha_1_beta_0_IB":
                ("alpha_beta", "alpha_beta", "alpha_beta", "alphabetax",
                 {"alpha": 1, "beta": 0, "ignore_bias": True}),
            "lrp.alpha_2_beta_1_IB":
                ("alpha_beta", "alpha_beta", "alpha_beta", "alphabetax",
                 {"alpha": 2, "beta": 1, "ignore_bias": True}),
            "lrp.z_plus":
                ("z+", "z+", "z+", "alphabetax", {}),
            "lrp.epsilon":
                ("epsilon", "epsilon", "epsilon", "alphabetax", None),
            "lrp.epsilon_IB":
                ("epsilon_IB", "epsilon_IB", "epsilon_IB", "alphabetax",
                 {"ignore_bias": True}),
            "deep_taylor.bounded":
                ("z+", "zB", "z+", "alphabetax", {}),
            "deep_taylor":
                ("z+", "z+", "z+", "alphabetax", {}),
            "lrp.alpha_beta":
                ("alpha_beta", "alpha_beta", "alpha_beta", "alphabetax", {}),
            "lrp.alpha_beta_eps":
                ("alpha_beta", "alpha_beta", "epsilon", "alphabetax", {}),
        }[lrp_config]
    return lrp_method, lrp_method_input, lrp_method_linear, lrp_method_batchnorm, lrp_params
