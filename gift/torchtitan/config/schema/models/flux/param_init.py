from torchtitan.config.schema.protocols.module import Normal, Ones, XavierUniform, Zeros


_ZERO_LINEAR = {"weight": Zeros(), "bias": Zeros()}
_XAVIER_LINEAR = {"weight": XavierUniform(), "bias": Zeros()}
_NORMAL_02 = {"weight": Normal(std=0.02), "bias": Zeros()}
_NORM_INIT = {"weight": Ones()}
