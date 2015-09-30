from scipy import random, array, argmax, cumsum, log
from datetime import datetime


class BernoulliArm(object):
    def __init__(self, p=0.5, payout=1.):
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
        return random.binomial(1, p=self.p) * self.payout


class MultinomialArm(object):
    def __init__(self, pvals=(0.5, 0.5), payouts=(0., 1.)):
        if len(pvals) != len(payouts):
            raise ValueError("pvals and payouts must have same length")

        self._pvals = None
        self.pvals = pvals
        self.payouts = payouts

    @property
    def pvals(self):
        return self._pvals

    @pvals.setter
    def pvals(self, new_pvals):
        if not all(i >= 0 for i in new_pvals):
            raise ValueError("All pvals must be non-negative")

        if sum(new_pvals) > 1:
            raise ValueError("All pvals must sum to at-most 1")

        self._pvals = array(new_pvals)

    @property
    def expected_reward(self):
        return self.pvals.dot(self.payouts)

    def draw(self):
        return random.multinomial(1, pvals=self.pvals).dot(self.payouts)


class BanditMonteCarloSimulation(object):
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