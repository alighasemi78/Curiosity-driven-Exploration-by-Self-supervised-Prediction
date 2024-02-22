from curiosity.base import Curiosity, CuriosityFactory


class NoCuriosity(Curiosity):
    def __init__(self):
        pass

    def reward(self, rewards, states, actions):
        return rewards

    def loss(self, policy_loss, states, next_states, actions):
        return policy_loss

    def parameters(self):
        yield from ()

    @staticmethod
    def factory():
        return NoCuriosityFactory()


class NoCuriosityFactory(CuriosityFactory):
    def create(self, state_converter, action_converter):
        return NoCuriosity()
