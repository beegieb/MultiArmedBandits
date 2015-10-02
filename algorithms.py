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
    SoftmaxBandit selects arms stochastically by creating a categorical distribution across arms via a softmax function
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
    """
    DirichletBandit selects arms stochastichally from a categorical distribution sampled from a Dirichlet distribution

    This bandit samples priors for the categorical distribution, and then randomly selects the arm from the given
    categorical distribution
    """
    def __init__(self, random_sample=True, sample_priors=True, **kwargs):
        """
        :param random_sample: a boolean (default True)
                              if True, the selected arm is drawn at random from a categorical distribution
                              if False, the argmax from categorical parameters is returned as the selected arm
        :param sample_priors: a boolean (default True)
                              if True, parameter for the categorical are sampled at random from a Dirichlet distribution
                              if False, parameters for the categorical are given by the mean of a Dirichlet distribution
        :param kwargs: Arguments to pass to BaseBandit superclass
        """
        self.random_sample = random_sample
        self.sample_priors = sample_priors
        super(DirichletBandit, self).__init__(**kwargs)

    def draw(self):
        """
        if sample_priors = True and random_sample = True:
           draw returns a random draw of a categorical distribution with parameters drawn from a Dirichlet distribution
           the hyperparameters on the Dirichlet are given by the bandit's metric with laplacian smoothing
        if sample_priors = False and random_sample = True:
            draw returns a random draw of a categorical distribution with parameters given by the bandit's metric
        if sample_priors = True and random_sample = False:
            draw returns argmax(random.dirichlet((x_0 + 1, ... , x_n_arms + 1))) where x_i is the ith value returned by
            the bandit's metric.
        if sample_priors = False and random_sample = False:
            become a purely greedy bandit with the selected arm given by argmax(metric)

        :return: The numerical index of the selected arm
        """
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
    """
    Nearly identical to the DirichletBandit, the only difference is annealing is applied when samping parameters from
    the Dirichlet Distribution. Annealing has the effect of reducing the variance in samples pulled from the Dirichlet
    distribution as the temperature decreases.
    """
    def __init__(self, random_sample=True, sample_priors=True, **kwargs):
        """
        :param random_sample: a boolean (default True)
                              if True, the selected arm is drawn at random from a categorical distribution
                              if False, the argmax from categorical parameters is returned as the selected arm
        :param sample_priors: a boolean (default True)
                              if True, parameter for the categorical are sampled at random from a Dirichlet distribution
                              if False, parameters for the categorical are given by the mean of a Dirichlet distribution
        :param kwargs: Arguments to pass to AnnealedBaseBandit superclass
        """
        self.random_sample = random_sample
        self.sample_priors = sample_priors
        super(AnnealedDirichletBandit, self).__init__(**kwargs)

    def draw(self):
        """
        if sample_priors = True and random_sample = True:
           draw returns a random draw of a categorical distribution with parameters drawn from a Dirichlet distribution
           the hyperparameters on the Dirichlet are given by the bandit's metric with laplacian smoothing
        if sample_priors = False and random_sample = True:
            draw returns a random draw of a categorical distribution with parameters given by the bandit's metric
        if sample_priors = True and random_sample = False:
            draw returns argmax(random.dirichlet((x_0 + 1, ... , x_n_arms + 1))) where x_i is the ith value returned by
            the bandit's metric.
        if sample_priors = False and random_sample = False:
            become a purely greedy bandit with the selected arm given by argmax(metric)

        :return: The numerical index of the selected arm
        """
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
    """
    An Upper Confidence Bound bandit that assumes each arm's chance of success is given by a Bernoulli distribution,
    and the payout of each arm is identical

    The bandit assumes the Bernoulli parameters are generated from a Beta prior whose uncertainty can be quantified

    Arms are selected deterministically by selecting the arm with the highest estimated upper confidence bound on
    the beta priors
    """
    def __init__(self, conf=0.95, **kwargs):
        """
        :param conf: The 2-sided confidence interval to use when calculating the Upper Confidence Bound (default 0.95)
        :param kwargs: Arguments to pass to BaseBandit superclass

        Note: metric is ignored in this bandit algorithm. The beta distribution parameters are given by success and
        failure rates of each individual arm
        """
        self.conf = conf
        super(UCBBetaBandit, self).__init__(**kwargs)

    def draw(self):
        """
        Selects the arm to draw based on the upper bounds of each arm's confidence interval

        Specifically returns: argmax([... beta(succ_i + 1, fail_i + 1).interval(conf) ... ])
        where succ_i and fail_i are the total number of successful and failed pulls for the ith arm

        :return: The numerical index of the selected arm
        """
        succ = array(self.success)
        fail = array(self.draws) - succ
        beta = stats.beta(succ + 1, fail + 1)

        return argmax(beta.interval(self.conf)[1])


class RandomBetaBandit(BaseBandit):
    """
    The RandomBetaBandit has similar assumptions to the UCBBetaBandit. But instead of estimating the probability of
    success for each arm by looking at the upper confidence bound, this bandit instead samples the probability of
    success for each arm from a beta distribution

    This has the effect of introducing randomness into the process of selecting arms, while accounting for uncertainty
    in the success rates of individual arms. There is also the added bonus that sampling is computationally faster
    than computing upper confidence bounds on a Beta distribution
    """
    def draw(self):
        """
        Selects the arm with the largest sampled probability of success

        Specifically returns: argmax([... random.beta(succ_i + 1, fail_i + 1) ... ])
        where succ_i and fail_i are the total number of successful and failed pulls for the ith arm

        :return: The numerical index of the selected arm
        """
        succ = array(self.success)
        fail = array(self.draws) - succ
        rvs = random.beta(succ + 1, fail + 1)

        return argmax(rvs)


class UCB1Bandit(BaseBandit):
    """
    Implements the UCB1 algorithm, one of the simplest in the UCB family of bandits.

    The implementation details can be found in the following publication:
        http://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    """
    def draw(self):
        """
        Draws arm based on the highest expected reward with a bonus given for uncertainty.

        Concretely:
            draws argmax([... expected_payout[i] + sqrt(2*log(T[i]) / draws[i]) ...])

        :return: The numerical index of the selected arm
        """
        t = 2*log(self.total_draws)

        return argmax([e + sqrt(t/d) if d > 0 else float('inf') for e, d in zip(self.expected_payouts, self.draws)])


class UCBGaussianBandit(BaseBandit):
    """
    UCBGaussianBandit is another UCB bandit that models expected payout for each arm as a univariate-gaussian
    distribution. The bandit selects the arm with the highest 95% confidence bound for expected reward, which is
    computed in closed form using the approximation:
        upper_bound[i] = mean[i] + 1.96 * std[i]

    This model uses an online algorithm for computing variance described on Wikipedia:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """

    def initialize(self, n_arms):
        """
        Initialize the bandit algorithm with lists for draws, payouts, success, and online variance

        :param n_arms: an int of the number of arms of the bandit
        """
        self.M2 = [0 for _ in range(n_arms)]
        super(UCBGaussianBandit, self).initialize(n_arms)

    def update(self, selected_arm, payout):
        """
        Update the bandits parameters by incrementing each of:
            draws[selected_arm], payouts[selected_arm], and success[selected_arm]

        Also updates tracking for online variance estimates

        :param selected_arm: an int on interval [0, n_arms)
        :param payout: the total payout recieved from selected_arm
        """
        delta = payout - self.expected_payouts[selected_arm]
        super(UCBGaussianBandit, self).update(selected_arm, payout)
        mean = self.expected_payouts[selected_arm]
        self.M2[selected_arm] += delta * (payout - mean)

    def draw(self):
        """
        If an arm has been drawn less than 2 times, select that arm

        Otherwise return:
            argmax([ ... expected_reward[i] + 1.96 * std[i] ...])

        :return: The numerical index of the selected arm
        """
        mu = self.expected_payouts
        M2 = self.M2
        counts = self.draws

        return argmax(float('inf') if n < 2 else m + 1.96 * sqrt(s / (n - 1)) for m, s, n in zip(mu, M2, counts))


class RandomGaussianBandit(UCBGaussianBandit):
    """
    Similar model to the UCBGaussianBandit, the difference being the model randomly samples the estimates for
    expected reward from the learned gaussians. This adds randomness the draws allowing the algorithm to better handle
    settings with delayed feedback.

    Some imperical tests also provide evidence that this algorithm outperforms the UCBGaussianBandit in settings with
    instantanious feedback, but this is not a proven fact. Use that observation with caution.
    """
    def draw(self):
        """
        If an arm has been drawn less than 2 times, select that arm

        Otherwise return:
            argmax([ ... random.normal(mean=expected_return[i], sd=std[i]) ...])

        :return: The numerical index of the selected arm
        """
        mu = array(self.expected_payouts)
        sd = array([float('inf') if n < 2 else sqrt(s / (n - 1)) for s, n in zip(self.M2, self.draws)])

        return argmax(random.randn(self.n_arms) * sd + mu)