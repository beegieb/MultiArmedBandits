from scipy import random, array, argmax, cumsum, log
from datetime import datetime


class BernoulliArm(object):
    """
    A BernoulliArm simulates an arm with a single payout with a fixed payout probability
    """
    def __init__(self, p=0.5, payout=1.):
        """
        :param p: The probability of outputting a payout (default: 0.5)
        :param payout: The numerical value of the payout (default: 1.0)
        """
        self._p = None
        self.p = p
        self.payout = payout

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p):
        if new_p > 1 or new_p < 0:
            raise ValueError('p of %f is not in the interval [0,1]' % new_p)

        self._p = new_p

    @property
    def expected_reward(self):
        return self.p * self.payout

    def draw(self):
        """
        :return: payout with probability p, 0 with probability (1 - p)
        """
        return random.binomial(1, p=self.p) * self.payout


class CategoricalArm(object):
    """
    CategoricalArm simulates an arm that has multiple different payout values each with probability of success
    given by a categorical distribution
    """
    def __init__(self, pvals=(0.5, 0.5), payouts=(0., 1.)):
        """
        :param pvals: a list-like of probabilities corresponding to each possible payout (default: (0.5, 0.5))
                      The condition must hold: sum(pvals) <= 1
                      if sum(pvals) < 1, the final element in pvals is renormalized to 1 - sum(pvals[:-1]) when
                      making a draw
        :param payouts: a list-like of payout values (default: (0., 1.))
                        must be same length as pvals
        """
        if len(pvals) != len(payouts):
            raise ValueError("pvals and payouts must have same length")

        self._pvals = None
        self._payouts = payouts
        self.pvals = pvals

    @property
    def pvals(self):
        return self._pvals

    @pvals.setter
    def pvals(self, new_pvals):
        if not all(i >= 0 for i in new_pvals):
            raise ValueError("All pvals must be non-negative")

        if sum(new_pvals) > 1:
            raise ValueError("All pvals must sum to at-most 1")

        if len(new_pvals) != len(self.payouts):
            raise ValueError("pvals must have the same length as payouts, got %i expected %i"
                             % len(new_pvals), len(self.payouts))

        self._pvals = array(new_pvals)

    @property
    def payouts(self):
        return self._payouts

    @payouts.setter
    def payouts(self, new_payouts):
        if len(new_payouts) != len(self.pvals):
            raise ValueError("payouts must have the same length as pvals, got %i expected %i"
                             % len(new_payouts), len(self.pvals))

        self._payouts = new_payouts

    @property
    def expected_reward(self):
        return self.pvals.dot(self.payouts)

    def draw(self):
        """
        Sample from the categorical distribution, and payout corresponding to the selected index

        :return: payouts[i], where i is sampled from the categorical distribution given by pvals
        """
        return random.multinomial(1, pvals=self.pvals).dot(self.payouts)


class BanditSimulation(object):
    def __init__(self, arms, n_rounds=500, n_sim=500, verbose=False, outfile=None):
        self.arms = arms
        self.n_rounds = n_rounds
        self.n_sim = n_sim
        self.outfile = outfile
        self.verbose = verbose
        self._round_counter = 0
        self._simulation_counter = 0

    @property
    def n_arms(self):
        return len(self.arms)

    def _run_one_round(self, bandit_alg):
        arm = bandit_alg.draw()
        payout = self.arms[arm].draw()
        bandit_alg.update(selected_arm=arm, payout=payout)

        self._round_counter += 1

        if self.verbose:
            print 'Round %i - Arm %i - Payout %f' % (self._round_counter, arm, payout)

        return arm, payout

    def _run_one_sim(self, bandit_alg):
        self._round_counter = 0
        random.shuffle(self.arms)
        bandit_alg.initialize(n_arms=self.n_arms)
        best_arm = argmax([arm.expected_reward for arm in self.arms])

        self._simulation_counter += 1

        if self.verbose:
            print 'Starting Simulation %i - Best Arm is %i' % (self._simulation_counter, best_arm)

        start_time = datetime.now()
        results = [self._run_one_round(bandit_alg) for _ in range(self.n_rounds)]
        end_time = datetime.now()

        return results, best_arm, (end_time - start_time).seconds

    def simulate(self, bandit_alg):
        self.results_ = [self._run_one_sim(bandit_alg) for _ in range(self.n_sim)]

        if self.outfile is not None:
            self.save_results(outfile=self.outfile)

        if self.verbose:
            print 'Done'

        return self

    def summary(self):
        if not hasattr(self, 'results_'):
            raise RuntimeError('Simulation has not been run yet. Use the simulate method prior to calling save_results')

        runs = array([run for run, _, _ in self.results_])
        best_arms = [best_arm for _, best_arm, _ in self.results_]
        runtimes = array([runtime for _, _, runtime in self.results_])

        cumulative_rewards = cumsum(runs[:, :, 1], 1)
        best_arm_accuracy = array([r[:, 0] == a for r, a in zip(runs, best_arms)])

        return {'Runtime': {'Avg': runtimes.mean(), 'Std': runtimes.std(), 'Total': runtimes.sum()},
                'Accuracy': {'Avg': best_arm_accuracy.mean(0)},
                'CumulativeRewards': {'Avg': cumulative_rewards.mean(0), 'Std': cumulative_rewards.std(0)}}

    def print_summary(self):
        summary = self.summary()
        n = len(summary['Accuracy']['Avg'])
        discount_weights = log(array(range(n)) + 1) + 1
        discount_weights /= sum(discount_weights)
        print 'Final Average Accuracy: %f' % summary['Accuracy']['Avg'][-1]
        print 'Discounted Average Accuracy: %f' % summary['Accuracy']['Avg'].dot(discount_weights)
        print 'Average Total Reward: %f - Std: %f' % (summary['CumulativeRewards']['Avg'][-1],
                                                      summary['CumulativeRewards']['Std'][-1])
        print 'Average Runtime: %f - Total Runtime: %f' % (summary['Runtime']['Avg'], summary['Runtime']['Total'])

    def save_results(self, outfile):
        if not hasattr(self, 'results_'):
            raise RuntimeError('Simulation has not been run yet. Use the simulate method prior to calling save_results')

        if self.verbose:
            print 'Saving simulation results to %s' % outfile

        with open(outfile, 'w') as output:
            output.write('Simulation_ID,Round,Arm,Payout,Best_Arm\n')
            for i, (sim_round, best_arm, runtime) in enumerate(self.results_):
                for j, (arm, payout) in enumerate(sim_round):
                    output.write('%i,%i,%i,%f,%i\n' % (i+1, j+1, arm, payout, best_arm))