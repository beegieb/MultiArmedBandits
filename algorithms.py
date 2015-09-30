from __future__ import division
from scipy import random, exp, log, sqrt, argmax, array, stats

TINY = 1e-6


class BaseBandit(object):
    def __init__(self, draws=None, payouts=None, success=None, n_arms=None, metric='payout'):
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

        self.metric = metric

    def initialize(self, n_arms):
        self.draws = [0]*n_arms
        self.payouts = [0]*n_arms
        self.success = [0]*n_arms

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, new_metric):
        if new_metric in {'Epayout', 'Esuccess', 'payout', 'success'}:
            self._metric = new_metric
        else:
            raise ValueError('metric must be either "payout", "success", "Epayout", or "Esuccess"')

    def _metric_fn(self):
        if self.metric == 'payout':
            return self.payouts

        elif self.metric == 'success':
            return self.success

        elif self.metric == 'Epayout':
            return self.expected_payouts

        elif self.metric == 'Esuccess':
            return self.expected_success

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
        return [s/d if d > 0 else 0 for s, d in zip(self.success, self.draws)]

    @property
    def expected_payouts(self):
        return [p/d if d > 0 else 0 for p, d in zip(self.payouts, self.draws)]

    def update(self, selected_arm, payout):
        self.draws[selected_arm] += 1
        self.payouts[selected_arm] += payout
        self.success[selected_arm] += 1 if payout > 0 else 0

    def draw(self):
        raise NotImplementedError('This is a baseclass, inherit this class and implement a "draw" method')


def linear_schedule(t):
    return t + TINY


def logarithmic_schedule(t):
    return log(t + 1 + TINY)


class AnnealedBaseBandit(BaseBandit):
    def __init__(self, schedule='logarithmic', **kwargs):
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
            return argmax(self._metric_fn())


class AnnealedEpsilonGreedyBandit(AnnealedBaseBandit):
    def __init__(self, epsilon=1.0, **kwargs):
        self.epsilon = epsilon
        super(AnnealedEpsilonGreedyBandit, self).__init__(**kwargs)

    def draw(self):
        temp = 1 / self._schedule_fn(self.total_draws)
        if random.rand() < self.epsilon * temp:
            return random.choice(self.n_arms)
        else:
            return argmax(self._metric_fn())


def softmax(l):
    ex = exp(array(l) - max(l))
    return ex / ex.sum()


class SoftmaxBandit(BaseBandit):
    def draw(self):
        return argmax(random.multinomial(1, pvals=softmax(self._metric_fn())))


class AnnealedSoftmaxBandit(AnnealedBaseBandit):
    def draw(self):
        temp = 1 / self._schedule_fn(self.total_draws)
        return argmax(random.multinomial(1, pvals=softmax(array(self._metric_fn()) / temp)))


class DirichletBandit(BaseBandit):
    def __init__(self, random_sample=True, sample_priors=True, **kwargs):
        self.random_sample = random_sample
        self.sample_priors = sample_priors
        super(DirichletBandit, self).__init__(**kwargs)

    def draw(self):
        x = array(self._metric_fn()) + 1

        if self.sample_priors:
            pvals = random.dirichlet(x)
        else:
            pvals = x / sum(x)

        if self.random_sample:
            return argmax(random.multinomial(1, pvals=pvals))
        else:
            return argmax(pvals)


class AnnealedDirichletBandit(AnnealedBaseBandit):
    def __init__(self, random_sample=True, sample_priors=True, **kwargs):
        self.random_sample = random_sample
        self.sample_priors = sample_priors
        super(AnnealedDirichletBandit, self).__init__(**kwargs)

    def draw(self):
        temp = 1 / self._schedule_fn(self.total_draws)
        x = array(self._metric_fn()) / temp + 1

        if self.sample_priors:
            pvals = random.dirichlet(x)
        else:
            pvals = x / sum(x)

        if self.random_sample:
            return argmax(random.multinomial(1, pvals=pvals))
        else:
            return argmax(pvals)


class UCBBetaBandit(BaseBandit):
    def __init__(self, conf=0.95, **kwargs):
        self.conf = conf
        super(UCBBetaBandit, self).__init__(**kwargs)

    def draw(self):
        succ = array(self.success)
        fail = array(self.draws) - succ
        beta = stats.beta(succ + 1, fail + 1)

        return argmax(beta.interval(self.conf)[1])


class RandomBetaBandit(BaseBandit):
    def draw(self):
        succ = array(self.success)
        fail = array(self.draws) - succ
        rvs = random.beta(succ + 1, fail + 1)

        return argmax(rvs)


class UCB1Bandit(BaseBandit):
    def draw(self):
        t = 2*log(self.total_draws)

        return argmax([e + sqrt(t/d) if d > 0 else 1 for e, d in zip(self.expected_payouts, self.draws)])


class UCBGaussianBandit(BaseBandit):
    def __init__(self, **kwargs):
        super(UCBGaussianBandit, self).__init__(**kwargs)

    def initialize(self, n_arms):
        self.M2 = [0 for _ in range(n_arms)]
        super(UCBGaussianBandit, self).initialize(n_arms)

    def update(self, selected_arm, payout):
        delta = payout - self.expected_payouts[selected_arm]
        super(UCBGaussianBandit, self).update(selected_arm, payout)
        mean = self.expected_payouts[selected_arm]
        self.M2[selected_arm] += delta * (payout - mean)

    def draw(self):
        mu = self.expected_payouts
        M2 = self.M2
        counts = self.draws

        return argmax(float('inf') if n < 2 else m + 1.96 * sqrt(s / (n - 1)) for m, s, n in zip(mu, M2, counts))


class RandomGaussianBandit(UCBGaussianBandit):
    def draw(self):
        mu = array(self.expected_payouts)
        sd = array([float('inf') if n < 2 else sqrt(s / (n - 1)) for s, n in zip(self.M2, self.draws)])

        return argmax(random.randn(self.n_arms) * sd + mu)