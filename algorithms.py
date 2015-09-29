from __future__ import division
from scipy import random, exp, log, argmax, array


TINY = 1e-8


class BaseBandit(object):
    def __init__(self, draws=None, payouts=None, success=None, n_arms=None):
        if draws is None or payouts is None or success is None:
            if n_arms is None:
                raise ValueError('Must give either draws, payouts, and success or n_arms')
            else:
                self.initialize(n_arms)
        else:
            if len(draws) != len(payouts) and len(draws) != len(success):
                raise ValueError('draws, payouts, and success must all have identical lengths')
            else:
                self.draws = draws
                self.payouts = payouts
                self.success = success

    def initialize(self, n_arms):
        self.draws = [0]*n_arms
        self.payouts = [0]*n_arms
        self.success = [0]*n_arms

    @property
    def total_draws(self):
        return sum(self.draws)

    @property
    def total_success(self):
        return sum(self.success)

    @property
    def total_payouts(self):
        return sum(self.payouts)

    @property
    def n_arms(self):
        return len(self.draws)

    @property
    def expected_success(self):
        return [s/d for s, d in zip(self.success, self.draws)]

    @property
    def expected_payouts(self):
        return [p/d for p, d in zip(self.payouts, self.draws)]

    def update(self, selected_arm, reward):
        self.draws[selected_arm] += 1
        self.payouts[selected_arm] += reward
        self.success[selected_arm] += 1 if reward > 0 else 0

    def draw(self):
        raise NotImplementedError('This is a baseclass, inherit this class and implement a "draw" method')


def linear_schedule(t):
    return t + TINY


def logarithmic_schedule(t):
    return log(t + 1 + TINY)


class AnnealedBaseBandit(BaseBandit):
    def __init__(self, schedule='linear', **kwargs):
        self.schedule = schedule
        super(AnnealedBaseBandit, self).__init__(**kwargs)

    @property
    def schedule(self):
        return self._schedule_name

    @schedule.setter
    def schedule(self, new):
        if new == 'linear':
            self._schedule_name = new
            self._schedule_fn = linear_schedule
        elif new == 'logarithmic':
            self._schedule_name = new
            self._schedule_fn = logarithmic_schedule
        else:
            raise ValueError('Incorrect value for annealing schedule. Got %s. Expected "linear" or "logarithmic"' % new)


class GreedyBandit(BaseBandit):
    def draw(self):
        return argmax(self.expected_payouts)


class EpsilonGreedyBandit(BaseBandit):
    def __init__(self, epsilon=0.1, **kwargs):
        self.epsilon = epsilon
        super(EpsilonGreedyBandit, self).__init__(**kwargs)

    def draw(self):
        if random.rand() < self.epsilon:
            return random.choice(self.n_arms)
        else:
            return argmax(self.expected_payouts)


class AnnealedEpsilonGreedyBandit(AnnealedBaseBandit):
    def __init__(self, epsilon=1.0, **kwargs):
        self.epsilon = epsilon
        super(AnnealedEpsilonGreedyBandit, self).__init__(**kwargs)

    def draw(self):
        temp = 1 / self._schedule_fn(self.total_draws)
        if random.rand() < self.epsilon * temp:
            return random.choice(self.n_arms)
        else:
            return argmax(self.expected_payouts)


def softmax(l):
    ex = exp(array(l) - max(l))
    return ex / ex.sum()


class SoftmaxBandit(BaseBandit):
    def draw(self):
        return argmax(random.multinomial(1, pvals=softmax(self.expected_payouts)))


class AnnealedSoftmaxBandit(AnnealedBaseBandit):
    def draw(self):
        temp = 1 / self._schedule_fn(self.total_draws)
        return argmax(random.multinomial(1, pvals=softmax(array(self.expected_payouts) / temp)))


class DirichletBandit(BaseBandit):
    def __init__(self, random_sample=True, **kwargs):
        self.random_sample = random_sample
        super(DirichletBandit, self).__init__(**kwargs)

    def draw(self):
        x = array(self.expected_payouts) + 1

        if self.random_sample:
            pvals = random.dirichlet(x)
        else:
            pvals = x / sum(x)

        return argmax(random.multinomial(1, pvals=pvals))