from __future__ import division
from scipy import random, exp, log, sqrt, argmax, array, stats

TINY = 1e-6


class BaseBandit(object):
    """
    Baseclass for Bandit Algorithms. This is intended to be inherited by other Bandits to provide core functions.

    The BaseBandit takes care of basic initialization, and update rules. The class also exposes a number of useful
    properties for tracking metrics useful for monitoring bandit algorithms.

    Properties and Attributes exposed by this baseclass:
        n_arms - the number of arms available to the bandit
        draws - the number of draws performed by the bandit for each arm
        payouts - the total payouts given to the algorithm for each arm
        success - the total number of successful payouts for each arm
        expected_payouts - the expected payout for each arm
        expected_success - the expected success rate of each arm
        total_draws - the total number of draws performed by the bandit
        total_payouts - the total payout achieved by the bandit
        total_success - the total number of successful draws achieved by the bandit
        metric - the type of performance metric to use when deciding on which arm to draw

    Additionally, the BaseBandit provides a 'hidden' function _metric_fn which exposes the relevent performance
    metric, as a list, to all subclasses
    """
    def __init__(self, draws=None, payouts=None, success=None, n_arms=None, metric='payout'):
        """
        Must supply either: draws, payouts AND success OR n_arms.

        If draws, payouts, AND success, each must have the same length.

        :param draws: None or a list containing the number of draws for each arm (default = None)
        :param payouts: None or a list containing the total payouts for each arm (default = None)
        :param success: None or a list containing the success counts for each arm (default = None)
        :param n_arms: None or an int of the number of arms of the bandit (default = None)
        :param metric: Either 'payout', 'success', 'Epayout', 'Esuccess' (default = 'payout')
                       Epayout, Esuccess stand for expected_payout and expected_success

                       This is the performance metric that will be exposed via BaseBandit._metric_fn
        """
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
        """
        Initialize the bandit algorithm with lists for draws, payouts, and success

        :param n_arms: an int of the number of arms of the bandit
        """
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
        """
        Update the bandits parameters by incrementing each of:
            draws[selected_arm], payouts[selected_arm], and success[selected_arm]

        :param selected_arm: an int on interval [0, n_arms)
        :param payout: the total payout recieved from selected_arm
        """
        self.draws[selected_arm] += 1
        self.payouts[selected_arm] += payout
        self.success[selected_arm] += 1 if payout > 0 else 0

    def draw(self):
        raise NotImplementedError('This is a baseclass, inherit this class and implement a "draw" method')


def linear_schedule(t):
    return 1 / (t + TINY)


def logarithmic_schedule(t):
    return 1 / log(t + 1 + TINY)


class AnnealedBaseBandit(BaseBandit):
    """
    A subclass of BaseBandit intended to be inherited by annealing bandit algorithms

    Exposes the property:
        schedule - the type of annealing schedule for temperature updates

    Exposes the hidden method:
        _schedule_fn which outputs the current temperature at the current iteration
    """
    def __init__(self, schedule='logarithmic', **kwargs):
        """
        :param schedule: either 'logarithmic' or 'linear' (default = 'logarithmic')
                         'logarithmic' schedule updates temperature(iter_t) = 1 / log(t + 1 + 1e-6)
                         'linear' schedule updates temperature(iter_t) = 1 / (t + 1e-6)
        :param kwargs: Arguments that will be passed to the superclass BaseBandit
        """
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


class EpsilonGreedyBandit(BaseBandit):
    """
    The EpsilonGreedyBandit greedily selects the arm with the highest performing metric with probability (1-epsilon)
    and selects any arm, uniformly at random, with probability epsilon
    """
    def __init__(self, epsilon=0.1, **kwargs):
        """
        :param epsilon: a float on the interval [0, 1] (default = 0.1)
                        explore arms with probability epsilon, and exploit with probability (1 - epsilon)
        :param kwargs: Arguments to pass to the BaseBandit superclass
        """
        self.epsilon = epsilon
        super(EpsilonGreedyBandit, self).__init__(**kwargs)

    def draw(self):
        """
        Draws the best arm with probability (1 - epsilon)
        Draws any arm at random with probility epsilon

        :return: The numerical index of the selected arm
        """
        if random.rand() < self.epsilon:
            return random.choice(self.n_arms)
        else:
            return argmax(self._metric_fn())


class AnnealedEpsilonGreedyBandit(AnnealedBaseBandit):
    """
    An annealed version of the EpsilonGreedyBandit.

    Epsilon decreases over time proportional to the temperature given by the annealing schedule

    This has the effect of pushing the algorithm towards exploitation as time progresses
    """
    def __init__(self, epsilon=1.0, **kwargs):
        """
        :param epsilon: float on the interval [0, 1] (default = 1.0)
        :param kwargs: Arguments to pass to AnnealedBaseBandit superclass
        """
        self.epsilon = epsilon
        super(AnnealedEpsilonGreedyBandit, self).__init__(**kwargs)

    def draw(self):
        """
        Draws the best arm with probability (1 - epsilon * temp)
        Draws any arm with probability epsilon * temp

        :return: The numerical index of the selected arm
        """
        temp = self._schedule_fn(self.total_draws)
        if random.rand() < self.epsilon * temp:
            return random.choice(self.n_arms)
        else:
            return argmax(self._metric_fn())


def softmax(l):
    ex = exp(array(l) - max(l))
    return ex / ex.sum()


class SoftmaxBandit(BaseBandit):
    """
    SoftmaxBandit selects arms stochastically by creating a multinomial distribution across arms via a softmax function
    """
    def draw(self):
        """
        Selects arm i with probability distribution given by the softmax:
            P(arm_i) = exp(metric_i) / Z

        Where Z is the normalizing constant:
            Z = sum(exp(metric_i) for i in range(n_arms))

        :return: The numerical index of the selected arm
        """
        return argmax(random.multinomial(1, pvals=softmax(self._metric_fn())))


class AnnealedSoftmaxBandit(AnnealedBaseBandit):
    """
    Annealed version of the SoftmaxBandit
    """
    def draw(self):
        """
        Selects arm i with probability distribution given by the softmax:
            P(arm_i) = exp(metric_i / temperature) / Z

        Where Z is the normalizing constant:
            Z = sum(exp(metric_i / temperature) for i in range(n_arms))

        :return: The numerical index of the selected arm
        """
        temp = self._schedule_fn(self.total_draws)
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
        temp = self._schedule_fn(self.total_draws)
        x = array(self._metric_fn()) * temp + 1

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