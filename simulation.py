from scipy import random, array, argmax, log
from datetime import datetime
from collections import deque
from itertools import chain
import pandas as pd
import ggplot as gg


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

    def __repr__(self):
        return "BernoulliArm(p=%s, payout=%s)" % (self.p, self.payout)

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

    def __repr__(self):
        return "CategoricalArm(pvals=%s, payouts=%s)" % (self.pvals, self.payouts)

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
    def __init__(self, arms, n_rounds=500, n_sim=500, delay=0, verbose=False, outfile=None):
        self.arms = arms
        self.n_rounds = n_rounds
        self.n_sim = n_sim
        self.delay = delay
        self.outfile = outfile
        self.verbose = verbose
        self._round_counter = 0
        self._simulation_counter = 0
        self._delayed_updates = deque()

    def __repr__(self):
        return "BanditSimulation(arms=%s arms, n_rounds=%s, n_sim=%s, delay=%s, verbose=%s, outfile=%s)" % (
            len(self.arms), self.n_rounds, self.n_sim, self.delay, self.verbose, self.outfile)

    @property
    def n_arms(self):
        return len(self.arms)

    def _run_one_round(self, bandit_alg):
        start_time = datetime.now()
        arm = bandit_alg.draw()
        payout = self.arms[arm].draw()

        if self.delay == 0:
            bandit_alg.update(selected_arm=arm, payout=payout)
        else:
            while self.delay < len(self._delayed_updates):
                bandit_alg.update(**self._delayed_updates.popleft())
            self._delayed_updates.append({'selected_arm': arm, 'payout': payout})

        self._round_counter += 1

        if self.verbose:
            print 'Round %i - Arm %i - Payout %f' % (self._round_counter, arm, payout)
        end_time = datetime.now()
        return arm, payout, (end_time - start_time).total_seconds()

    def _run_one_sim(self, bandit_alg):
        self._round_counter = 0
        random.shuffle(self.arms)
        bandit_alg.initialize(n_arms=self.n_arms)
        best_arm = argmax([arm.expected_reward for arm in self.arms])

        self._simulation_counter += 1

        if self.verbose:
            print 'Starting Simulation %i - Best Arm is %i' % (self._simulation_counter, best_arm)

        results = [self._run_one_round(bandit_alg) for _ in range(self.n_rounds)]

        return results, best_arm

    def simulate(self, bandit_alg):
        results = [self._run_one_sim(bandit_alg) for _ in range(self.n_sim)]
        arm_runtime = pd.DataFrame({'simulation': i, 'best_arm': a} for i, (_, a) in enumerate(results))
        run_results = pd.DataFrame([{'selected_arm': r[0], 'payout': r[1], 'round': i % self.n_rounds,
                                     'runtime': r[2], 'simulation': i / self.n_rounds}
                                    for i, r in enumerate(chain(*(r[0] for r in results)))])
        self.results_ = run_results.join(arm_runtime, on='simulation',
                                         how='inner', lsuffix='_l').drop('simulation_l', axis=1)


        if self.outfile is not None:
            self.save_results(outfile=self.outfile)

        if self.verbose:
            print 'Done'

        return self

    def summary(self):
        if not hasattr(self, 'results_'):
            raise RuntimeError('Simulation has not been run yet. Use the simulate method prior to calling save_results')

        self.results_['is_best'] = self.results_.selected_arm == self.results_.best_arm
        grouper_sim = self.results_.groupby('simulation')
        grouper_round = self.results_.groupby('round')

        runtime = grouper_sim.runtime.sum()
        cumulative_payout = pd.DataFrame({'cumulative_payout': grouper_sim.payout.cumsum(),
                                           'round': range(self.n_rounds) * self.n_sim}).groupby('round')
        return {'Runtime': {'Avg': runtime.mean(), 'Std': runtime.std(), 'Total': runtime.sum()},
                'Accuracy': {'Avg': grouper_round.is_best.mean()},
                'CumulativePayout': {'Avg': cumulative_payout.mean().cumulative_payout,
                                     'Std': cumulative_payout.std().cumulative_payout}}

    def print_summary(self):
        summary = self.summary()
        n = len(summary['Accuracy']['Avg'])
        discount_weights = log(array(range(n)) + 1) + 1
        discount_weights /= sum(discount_weights)
        print 'Final Average Accuracy: %f' % summary['Accuracy']['Avg'].ix[self.n_rounds - 1]
        print 'Discounted Average Accuracy: %f' % summary['Accuracy']['Avg'].dot(discount_weights)
        print 'Average Total Reward: %f - Std: %f' % (summary['CumulativePayout']['Avg'].ix[self.n_rounds - 1],
                                                      summary['CumulativePayout']['Std'].ix[self.n_rounds - 1])
        print 'Average Runtime: %f - Total Runtime: %f' % (summary['Runtime']['Avg'], summary['Runtime']['Total'])

    def save_results(self, outfile, float_format='%.6f'):
        if not hasattr(self, 'results_'):
            raise RuntimeError('Simulation has not been run yet. Use the simulate method prior to calling save_results')

        if self.verbose:
            print 'Saving simulation results to %s' % outfile

        self.results_.to_csv(outfile, index=False, float_format=float_format)

    def _plot_cumulative_payouts(self, include_ci=True, summary=None):
        if summary is None:
            summary = self.summary()

        df = pd.DataFrame({'AverageCumulativePayout': summary['CumulativePayout']['Avg'],
                           'Std': summary['CumulativePayout']['Std'],
                           'Round': range(self.n_rounds)})
        if include_ci:
            df['ymin'] = df.AverageCumulativePayout - 1.96 * df.Std
            df['ymax'] = df.AverageCumulativePayout + 1.96 * df.Std
            plt = gg.ggplot(gg.aes(x='Round', y='AverageCumulativePayout', ymin='ymin', ymax='ymax'), data=df) + \
                  gg.geom_area(alpha=0.5)
        else:
            plt = gg.ggplot(gg.aes(x='Round', y='AverageCumulativePayout'), data=df)

        return plt + gg.geom_line()

    def _plot_avg_accuracy(self, include_ci=True, summary=None):
        if summary is None:
            summary = self.summary()

        df = pd.DataFrame({'AverageAccuracy': summary['Accuracy']['Avg'], 'Round': range(self.n_rounds)})

        if include_ci:
            from scipy import stats
            succ = df.AverageAccuracy * self.n_sim
            fail = self.n_sim - succ
            interval = stats.beta(succ + 1, fail + 1).interval(0.95)

            df['ymin'] = interval[0]
            df['ymax'] = interval[1]
            plt = gg.ggplot(gg.aes(x='Round', y='AverageAccuracy', ymin='ymin', ymax='ymax'), data=df) + \
                gg.geom_area(alpha=0.5)
        else:
            plt = gg.ggplot(gg.aes(x='Round', y='AverageAccuracy'), data=df)

        return plt + gg.geom_line()

    def plot(self, what='cumulative_payouts', include_ci=True):
        if what == 'cumulative_payouts':
            plt = self._plot_cumulative_payouts(include_ci=include_ci)
        elif what == 'avg_accuracy':
            plt = self._plot_avg_accuracy(include_ci=include_ci)
        elif what == 'all':
            summary = self.summary()
            p1 = self._plot_cumulative_payouts(include_ci=include_ci, summary=summary)
            p2 = self._plot_avg_accuracy(include_ci=include_ci, summary=summary)
            d1 = p1.data
            d2 = p2.data
            d1['Outcome'] = d1['AverageCumulativePayout']
            d2['Outcome'] = d2['AverageAccuracy']
            d1['Plot'] = 'Cumulative Payouts'
            d2['Plot'] = 'Average Accuracy'
            df = d1.append(d2, ignore_index=True)

            if include_ci:
                plt = gg.ggplot(gg.aes(x='Round', y='Outcome', ymin='ymin', ymax='ymax'), data=df) + \
                    gg.geom_area(alpha=0.5)
            else:
                plt = gg.ggplot(gg.aes(x='Round', y='Outcome'), data=df)
        else:
            raise ValueError('%s is not a valid option' % what)

        return plt + gg.geom_line() + gg.facet_grid('Plot', scales='free')