from pymc.distributions.distribution import NoDistribution
from pymc.distributions.mixture import Mixture



class DPBase(NoDistribution):
    """
    Base class for Dirichlet Process class
    """

    def __new__(
        cls, 
        name, 
        alpha, 
        G0, 
        **kwargs
    ):
        

        return super().__new__(cls, name, alpha, G0, **kwargs)

    def dist(cls, *params, **kwargs):
        return super().dist(params, **kwargs)

    def logp(x, *inputs):
        raise NotImplementedError


class DPMixture(NoDistribution):
    """
    Dirichlet Process Mixture
    """
    def __new__(cls, name, w, comp_dists, **kwargs):
        pass


class DPMixtureNormal(NoDistribution):
    """
    Class for a Dirichlet Process Mixture of Normals

    Should somehow inherit DPMixture
    """
    def __new__(
        cls,
        name,
        alpha,
        G0,
    )
