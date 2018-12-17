#!/usr/bin/python3

"""
Thu 13 Dec 2018 09:04:31 AM PST

@author:    aaron heuser
@version:   3.1
@filename:  system_dynamics_manager.py
"""


import ast
import csv
import itertools
import json
import numpy as np
import os
import scipy.integrate
import scipy.optimize


class SystemDynamicsManager():
    """
    Class used to keep track of all stocks and flows, as well as current and
    historical counts.
    """

    def __init__(self, params):
        """
        Parameters: params: dict
                        The dictionary of parameters, with keys: years, steps,
                        foi, num_foi, data_dir, init_fp, stochastic, sruns,
                        and gui. Within the data folder should the following
                        files:
                            addiction_down.json,
                            addiction_up.json,
                            cessation.json,
                            cessation_stochastic.json,
                            death.json,
                            health.json,
                            health_stochastic.json,
                            initial_counts.json,
                            initiation.json,
                            initiation_stochastic.json,
                            races.csv,
                            relapse.json,
                            relapse_stochastic.json,
                            stock_counts.csv
        Returns:    None
        """

        # Set the data folder path.
        self._data_dir = params['data_dir']
        # The read-write properties (contained in params).
        self._years = params['years']
        self._steps = params['steps']
        self._foi_param = params['foi']
        if os.path.isfile(self.gen_fp(params['init_fp'])):
            self._init_fp = params['init_fp']
        else:
            self._init_fp = None
        self._scheck = bool(params['stochastic'])
        if self.scheck:
            self._sruns = params['sruns']
        else:
            self._sruns = 1
        self._gui = params['gui']
        # The time steps.
        self._t = np.linspace(0, self.years, self.steps * self.years + 1)
        # The total US population.  
        self._us_pop_total = 325120000
        # The count of stocks for each (sex, race) combination.
        self._size = self.import_stock_counts()
        # Import the rate parameters. This will set self._lambda, self._chi,
        # self._gamma, self._kappa, self._death, self._phi, and self._psi.
        self.import_rate_data()
        # Import and set demographics data. This will set self._us_pop,
        # self._age, self._races, self._nraces, self._race_propse.
        self.set_demographics()
        # The number of FOI values to consider.
        self._num_foi = params['num_foi']
        # The range of foi values.
        self._foi_range = range(1, self.num_foi + 1)
        # Set the column heads for the output.
        self.set_output_columns()
        # Generate the influence distribution. This will set self._foi_dist and
        # self._exp_foi.
        self.gen_influence_distribution()
        # Import the initial stock counts. This will set self._init_counts.
        self.import_init_counts()
        # Convert the counts into densities. This will set self._y0.
        self.gen_initial_densities()
        # Set the transition rates for each stock. This will set self._rates.
        self.gen_stock_rates()
        # Generate the death rates ordered according the MFSG differential
        # system. This will set self._dy_death.
        self.gen_dy_death()
        # Generate an attribute with the size of each stock type. This will set
        # self._stock_size.
        self.gen_stock_size()
        # Generate a the canonical list of all stock keys. This will set the
        # attribute self._canonical_list.
        self.gen_canonical_list()
        # Generate the indices needed for estimating theta. This will set the
        # attribute self._theta_indices.
        self.gen_theta_indices()
        # Import the theta values (if they exist).
        self.import_theta()
        # Set the dictionary that will keep track of the solutions.
        self.mfsg = {}
        # Run the simulation.
        self.run()

    def gen_canonical_list(self):
        """
        Parameters: None
        Returns:    None
        """

        # The keys associated to S stocks.
        c_list = [(0, 0, 0, 0, 0, v) for v in self.product('S')]
        rows = ['S({0})'.format(x) for x in self.product('S')]
        # The keys associated to I stocks.
        c_list += [(0, 0, z, w, 0, v) for (z, w, v) in self.product('I')]
        rows += ['I{0}'.format(x) for x in self.product('I')]
        # The keys associated to U stocks.
        c_list += [(x, y, 0, 0, a, v) for (x, y, a, v) in self.product('U')]
        rows += ['U{0}'.format(x) for x in self.product('U')]
        # Finally, the keys for the Z stocks.
        c_list += list(self.product('Z'))
        rows += ['Z{0}'.format(x) for x in self.product('Z')]
        self._canonical_list = c_list
        self._rows = rows

    def gen_counts(self):
        """
        Parameters: None
        Returns:    None
        """

        counts = {}
        counts_low = {}
        counts_high = {}
        total = sum(sum(x) for x in self.init_counts.values())
        for key, val in self.mfsg.items():
            counts[key] = (total * val.sol).astype(int)
            if self.scheck:
                counts_low[key] = (total * val.sol_low).astype(int)
                counts_high[key] = (total * val.sol_high).astype(int)
        self._counts = counts
        self._counts_low = counts_low
        self._counts_high = counts_high

    def gen_death_counts(self):
        """
        Parameters: None
        Returns:    death: dict
                        The dictionary of (sex, race) to stock death counts.
        """

        death = {}
        death_low = {}
        death_high = {}
        for key, val in self.counts.items():
            dy_death = np.array(self.dy_death[key])
            death[key] = (dy_death * val.T).T.astype(int)
        if self.scheck:
            for key, val in self.counts_low.items():
                dy_death = np.array(self.dy_death[key])
                death_low[key] = (dy_death * val.T).T.astype(int)
            for key, val in self.counts_high.items():
                dy_death = np.array(self.dy_death[key])
                death_high[key] = (dy_death * val.T).T.astype(int)
        self._death_counts = death
        self._death_counts_low = death_low
        self._death_counts_high = death_high

    def gen_dy_death(self):
        """
        Parameters: None
        Returns:    None
        """

        # We generate a list of death rates with respect to the ordering of the
        # differential system (defined in the solution generator).
        death = {}
        # Set the ranges for each stock type. 
        for sr in self.product((2, self.nraces)):
            rng_S = self.product('S')
            rng_I = self.product('I')
            rng_U = self.product('U')
            rng_Z = self.product('Z')
            d = self.death[sr]
            death[sr] = []
            # We begin with the death rates for S stocks.
            death[sr] += [d[(0, 0, 0, 0, 0, v)] for v in rng_S]
            # Now for I stocks.
            death[sr] += [d[(0, 0, z, w, 0, v)] for (z, w, v) in rng_I]
            # The U stocks.
            death[sr] += [d[(x, y, 0, 0, a, v)] for (x, y, a, v) in rng_U]
            # Finally, the Z stocks.
            death[sr] += [d[key] for key in rng_Z]
        # Set the property self._dy_death.
        self._dy_death = death

    def gen_fp(self, fname):
        """
        Parameters: fname: str
                        The name of a file in the data folder.
        Returns:    fp: str
                        The path to the file.
        """

        fp = './{0}/{1}'.format(self.data_dir, fname)
        return fp

    def gen_influence_distribution(self):
        """
        Parameters: None
        Returns:    None
        """

        # We consider the branching distribution P(a) = c * a ** (-2 - u),
        # where u is the foi parameter. We thus determine the value of c.
        rng = self.foi_range
        c = 1 / sum(x ** (-2 - self.foi_param) for x in rng)
        # Using the constant, we can generate the distribution.
        self._foi_dist = [c * x ** (-2 - self.foi_param) for x in rng]
        self._exp_foi = np.dot(rng, self.foi_dist)

    def gen_initial_densities(self):
        """
        Parameters: None
        Returns:    None
        """

        y0 = {}
        # For each (sex, race) key in self.init_counts, we have an array of
        # counts. The respective densities are found by dividing by the sum of
        # this array.
        for key, val in self.init_counts.items():
            y0[key] = np.array([])
            for x in self.foi_range:
                x_arr = val[(x - 1) * self.size: x * self.size]
                x_sum = np.sum(x_arr)
                if x_sum > 0:
                    x_arr = x_arr / x_sum
                y0[key] = np.append(y0[key], x_arr)
        self._y0 = y0

    def gen_stock_rates(self):
        """
        Parameters: None
        Returns:    None
        """

        # For each stock in the system, we generate the transition rates,
        # separated by (sex, race) pairs and stock type S, I, U, Z.
        rates = {}
        for sr in self.product((2, self.nraces)):
            rates[sr] = {}
            # We begin with the susceptible stocks, which allow for transitions
            # due to initiation (lambda), infection (kappa), and aging (age).
            rates[sr]['S'] = {'lambda':{}, 'kappa':{}, 'age':{}}
            # Loop through each age group v and assign stock rates for S[v].
            for v in range(self.sizes['age_groups']):
                # We generate a list of all possible probabilities.
                q = [self.lambdas[sr][v], self.kappa[sr][(0, 0, v)]]
                # If we are not in the oldest age group, aging is allowed.
                if v < 2:
                    q += [np.array([self.age[v]])]
                # The death probability.
                d = self.death[sr][(0, 0, 0, 0, 0, v)]
                # Given the death rate and the possible transition rates, we
                # can determine the stock transition probabilities.
                trans = self.gen_transitions(q, d)
                rates[sr]['S']['lambda'][v] = trans[0]
                rates[sr]['S']['kappa'][v] = trans[1]
                if v < 2:
                    rates[sr]['S']['age'][v] = trans[2]
            # Now the transition rates for stocks of type I, which again allows
            # for initiation, infection, and aging.
            rates[sr]['I'] = {'lambda':{}, 'kappa':{}, 'age':{}}
            for (z, w, v) in self.product('I'):
                # For (z, w) we have the following possibilities (0, 1), (1,
                # 0), (1, 1), which influence the list of all possible
                # probabilities.
                q = [self.lambdas[sr][v]]
                # Since the resulting elements of q can vary in meaning, we
                # keep track of the index keys.
                q_keys = ['lambda']
                if (z, w) == (0, 1):
                    # In this case we can only make a health transition to
                    # condition one.
                    q += [self.kappa[sr][(0, 0, v)][0:1]]
                    q_keys += ['kappa']
                elif (z, w) == (1, 0):
                    # Now we can transition to condition two.
                    q += [self.kappa[sr][(0, 0, v)][1:2]]
                    q_keys += ['kappa']
                # No health transitions can occur if (z, w) == (1, 1), so we
                # proceed with possible age transitions.
                if v < 2:
                    q += [np.array([self.age[v]])]
                    q_keys += ['age']
                # The death probability.
                d = self.death[sr][(0, 0, z, w, 0, v)]
                # Generate the transition rates.
                trans = self.gen_transitions(q, d)
                # Now allocate to rates.
                for idx, key in enumerate(q_keys):
                    rates[sr]['I'][key][(z, w, v)] = trans[idx]
            # We now handle transition rates for stocks of type U, which allows
            # for initiation, infection, relapse, cessation, addiction, and
            # aging. We first define the parameter dictionary.
            keys = ['lambda', 'kappa', 'gamma', 'chi', 'phi', 'psi', 'age']
            rates[sr]['U'] = {x:{} for x in keys}
            for (x, y, a, v) in self.product('U'):
                # We have eight possible cases for (x, y): (0, 1), (0, 2), (1,
                # 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2).
                q = []
                q_keys = []
                # Initiation can occur only if x == 0 or y == 0. Note that
                # since individuals in U stocks are using at least one product,
                # we cannot have x == y == 0.
                if x == 0:
                    # We can initiate product 1 to become a dual user.
                    q += [self.lambdas[sr][v][0:1]]
                    q_keys += ['lambda']
                elif y == 0:
                    # We can initiate product 2 to become a dual user.
                    q += [self.lambdas[sr][v][1:2]]
                    q_keys += ['lambda']
                # Since individuals in U are not afflicted with either
                # condition, we can transition to both.
                q += [self.kappa[sr][(x, y, v)]]
                q_keys += ['kappa']
                # Relapse can only happen if x == 2 or y == 2.
                if (x, y) == (2, 2):
                    # In this case it is possible to relapse to either.
                    q += [self.gamma[sr][(a, v)]]
                    q_keys += ['gamma']
                elif x == 2:
                    # We can relapse to product 1.
                    q += [self.gamma[sr][(a, v)][0:1]]
                    q_keys += ['gamma']
                elif y == 2:
                    # We can relapse to product 2.
                    q += [self.gamma[sr][(a, v)][1:2]]
                    q_keys += ['gamma']
                # Cessation can occur only when x == 1 or y == 1.
                if (x, y) == (1, 1):
                    # We can cease either product.
                    q += [self.chi[sr][(a, v)]]
                    q_keys += ['chi']
                elif x == 1:
                    # We can cease product 1.
                    q += [self.chi[sr][(a, v)][0:1]]
                    q_keys += ['chi']
                elif y == 1:
                    # We can cease product 2.
                    q += [self.chi[sr][(a, v)][1:2]]
                    q_keys += ['chi']
                # We can increase the level of addiction when a < 3 and at
                # least one product is currently being used.
                if a < 3 and (x == 1 or y == 1):
                    q += [np.array([self.phi[a]])]
                    q_keys += ['phi']
                # We can decrease addiction only if we have some level of
                # addiction and are not using any products.
                if a > 0 and (x != 1 and y != 1):
                    q += [np.array([self.psi[a - 1]])]
                    q_keys += ['psi']
                # We can age if not in the oldest age group.
                if v < 2:
                    q += [np.array([self.age[v]])]
                    q_keys += ['age']
                # The death rate.
                d = self.death[sr][(x, y, 0, 0, a, v)]
                # Get the transition probabilities.
                trans = self.gen_transitions(q, d)
                # Reallocate to the rates.
                for idx, key in enumerate(q_keys):
                    rates[sr]['U'][key][(x, y, a, v)] = trans[idx]
            # We now handle transition rates for stocks of type Z, which allows
            # for initiation, infection, relapse, cessation, addiction, and
            # aging. We first define the parameter dictionary.
            keys = ['lambda', 'kappa', 'gamma', 'chi', 'phi', 'psi', 'age']
            rates[sr]['Z'] = {x:{} for x in keys}
            for (x, y, z, w, a, v) in self.product('Z'):
                # We have eight possible cases for (x, y): (0, 1), (0, 2), (1,
                # 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), and three for (z,
                # w): (0, 1), (1, 0), (1, 1).
                q = []
                q_keys = []
                # Initiation can occur only if x == 0 or y == 0. Note that
                # since individuals in Z stocks are using at least one product,
                # we cannot have x == y == 0.
                if x == 0:
                    # We can initiate product 1 to become a dual user.
                    q += [self.lambdas[sr][v][0:1]]
                    q_keys += ['lambda']
                elif y == 0:
                    # We can initiate product 2 to become a dual user.
                    q += [self.lambdas[sr][v][1:2]]
                    q_keys += ['lambda']
                # Health transitions are allowed when (z, w) != (1, 1).
                if z == 0:
                    # We can develop condition 1.
                    q += [self.kappa[sr][(x, y, v)][0:1]]
                    q_keys += ['kappa']
                elif w == 0:
                    # We can develop condition 2.
                    q += [self.kappa[sr][(x, y, v)][1:2]]
                    q_keys += ['kappa']
                # Relapse can only happen if x == 2 or y == 2.
                if (x, y) == (2, 2):
                    # In this case it is possible to relapse to either.
                    q += [self.gamma[sr][(a, v)]]
                    q_keys += ['gamma']
                elif x == 2:
                    # We can relapse to product 1.
                    q += [self.gamma[sr][(a, v)][0:1]]
                    q_keys += ['gamma']
                elif y == 2:
                    # We can relapse to product 2.
                    q += [self.gamma[sr][(a, v)][1:2]]
                    q_keys += ['gamma']
                # Cessation can occur only when x == 1 or y == 1.
                if (x, y) == (1, 1):
                    # We can cease either product.
                    q += [self.chi[sr][(a, v)]]
                    q_keys += ['chi']
                elif x == 1:
                    # We can cease product 1.
                    q += [self.chi[sr][(a, v)][0:1]]
                    q_keys += ['chi']
                elif y == 1:
                    # We can cease product 2.
                    q += [self.chi[sr][(a, v)][1:2]]
                    q_keys += ['chi']
                # We can increase the level of addiction when a < 3 and at
                # least one product is currently being used.
                if a < 3 and (x == 1 or y == 1):
                    q += [np.array([self.phi[a]])]
                    q_keys += ['phi']
                # We can decrease addiction only if we have some level of
                # addiction and are not using any products.
                if a > 0 and (x != 1 and y != 1):
                    q += [np.array([self.psi[a - 1]])]
                    q_keys += ['psi']
                # We can age if not in the oldest age group.
                if v < 2:
                    q += [np.array([self.age[v]])]
                    q_keys += ['age']
                # The death rate. 
                d = self.death[sr][(x, y, z, w, a, v)]
                # Get the transition probabilities.
                trans = self.gen_transitions(q, d)
                # Reallocate to the rates.
                for idx, key in enumerate(q_keys):
                    rates[sr]['Z'][key][(x, y, z, w, a, v)] = trans[idx]
        # Set the system stock transition rates.
        self._rates = rates

    def gen_stock_size(self):
        """
        Parameters: None
        Returns:    None
        """

        stock_size = {}
        for stock in ['S', 'I', 'U', 'Z']:
            stock_size[stock] = len(list(self.product(stock)))
        # Set the class property.
        self._stock_size = stock_size

    def gen_theta_file(self):
        """
        Parameters: None
        Returns:    None
        """

        # The file path for the theta file.
        fp = self.gen_fp('theta.json')
        # Check if the file exists, and generate if needed.
        if not os.path.isfile(fp):
            # Define the data dictionary class attribute and for export.
            theta = {}
            theta_export = {}
            for key, val in self.mfsg.items():
                theta[key] = val.theta
                theta_export[str(key)] = val.theta
            self._theta = theta
            with open(fp, 'w') as f:
                json.dump(theta_export, f, indent=0)

    def gen_theta_indices(self):
        """
        Parameters: None
        Returns:    None
        """

        # Generate the set of indices associated to theta contributing stocks.
        # The result will be set as self.theta_indices, a dictionary with keys
        # by the FOI level alpha and values given by lists. Element x of each
        # list contains the associated indices in the canonical stock list
        # (over all alpha), for product x.
        idxs = {}
        n_prods = self.sizes['tobacco_products']
        n_conds = self.sizes['health_conditions']
        # Loop through each alpha.
        for alpha in self.foi_range:
            idxs[alpha] = [[] for _ in range(n_prods)]
            # Loop through each key for 'U' stocks, and assign counts.
            for key in self.product('U'):
                # Set the full user key by including zeros for each health
                # condition considered.
                full_key = key[:n_prods] + tuple(n_conds * [0]) + key[n_prods:]
                # Loop through each product and append the stock density if
                # this key implies use of the given product.
                for p in range(n_prods):
                    if key[p] == 1:
                        # In this case the stock consists of users of product
                        # 1 (at least).
                        idxs[alpha][p] += [self.stock_index(full_key, alpha)]
            # Repeat for the 'Z' stocks.
            for key in self.product('Z'):
                # Loop through each product.
                for p in range(n_prods):
                    if key[p] == 1:
                        idxs[alpha][p] += [self.stock_index(key, alpha)]
        self._theta_indices = idxs

    def gen_transitions(self, q, d):
        """
        Parameters: q: list
                        A list of arrays, where each array gives the original
                        transition rates, independent of the system and other
                        transitions.
                    d: float
                        The probability of death.
        Returns:    p: list
                        The length len(q) list, where p[x] is the array
                        such that p[x][y] is respective transition probability.
        """

        # We need to flatten q into one numpy array.
        q_flat = np.concatenate(q)
        # Values will first be placed in a flattened array, with will be
        # expanded prior to being returned.
        p_flat = np.zeros(len(q_flat))
        # For a collection of Bernoulli random variables with respective
        # probabilites given by q, the possible outcomes can be described by 
        # a = (a_0, ..., a_n-1), where each a_x is either 0 or 1. The set of
        # outcomes that implies a transition to state x is all those such that
        # a_x == 1. If any other element a_y is also equal to 1, then it is
        # possible to transition to state y as well. Therefore, we multiple by
        # weights that equivalent to the probability that the given event
        # occurs prior to the others, under the assumption of exponential
        # waiting times with the given rates as parameters. For example, if we
        # have a = (1, 0, 1), then we have three possible states of transition,
        # and in this case, we could have a transition to state 0 or to state
        # 2. The respective weights would then be q[0] / (q[0] + q[2]) and
        # q[2] / (q[0] + q[2]). Begin by generating the sample space.
        ss_range = itertools.product(range(2), repeat=len(q_flat))
        ss = (np.array(x) for x in ss_range if sum(x) > 0)
        for state in ss:
            # Absent of the weights, each possible state transition will have
            # the same product term.
            prod_term_0 = q_flat * state
            prod_term_1 = (1 - q_flat) * (1 - state)
            trans = (1 - d) * np.prod(prod_term_0 + prod_term_1)
            # The weights determine the final transition values.
            weights = (q_flat * state) / np.sum(q_flat * state)
            # Since the weights are non-zero only when a contribution is given
            # to the respective probability, we need only add this to the
            # current list of probabilities.
            p_flat += weights * trans
        # Ensure stock probabilities sum to the proper value (approximately).
        p_sum = np.sum(p_flat) + d + (1 - d) * np.prod(1 - q_flat)
        if np.abs(p_sum - 1) > 0.000001:
            msg = 'Stock probabilities sum to {0}.'.format(np.sum(p_flat))
            raise ValueError(msg)
        # We now generate p by converting to the original shape of q.
        p = []
        idx_0 = 0
        for x in range(len(q)):
            p += [p_flat[idx_0: idx_0 + len(q[x])]]
            idx_0 += len(q[x])
        return p

    def import_init_counts(self):
        """
        Parameters: None
        Returns:    None
        """

        # Check first to see if we need to generate or import.
        counts = {}
        if self.init_fp is None:
            # We need a collection of counts such that the total count is equal
            # to the US population size.
            for (s, r) in self.product((2, self.nraces)):
                # We divide the population into the susceptible stocks. 
                total = int(self.us_pop[(s, r)] * self.us_pop_total)
                # We use rough estimates for age breakup found from the US
                # census (2015).
                counts_0 = int(total / self.size) * np.ones(self.size)
                # The int operation will round down, so we append the
                # discrepancy to the first stock.
                counts_0[0] += max(total - np.sum(counts_0), 0)
                counts_0 = counts_0.astype(int)
                counts[(s, r)] = counts_0
        else:
            # In this case we need to import the file of counts.
            with open(self.gen_fp('initial_counts.json'), 'r') as f:
                data_str = json.load(f)
            # Replace string keys in data with proper keys.
            data = {}
            for key_0, val_0 in data_str.items():
                k_0 = ast.literal_eval(key_0)
                data[k_0] = {}
                for key_1, val_1 in val_0.items():
                    k_1 = ast.literal_eval(key_1)
                    data[k_0][k_1] = val_1
            for (s, r) in self.product((2, self.nraces)):
                # Begin with counts for S.
                counts[(s, r)] = []
                for v in self.product('S'):
                    counts[(s, r)] += [data[(s, r)][(0, 0, 0, 0, 0, v)]]
                # The counts for I.
                for (z, w, v) in self.product('I'):
                    counts[(s, r)] += [data[(s, r)][(0, 0, z, w, 0, v)]]
                # The counts for U.
                for (x, y, a, v) in self.product('U'):
                    counts[(s, r)] += [data[(s, r)][(x, y, 0, 0, a, v)]]
                # Finally, the counts for Z.
                for key in self.product('Z'):
                    counts[(s, r)] += [data[(s, r)][key]]
                counts[(s, r)] = np.array(counts[(s, r)])
        # Having now the counts, we want to separate by influence level.
        counts_by_alpha = {}
        for key, val in counts.items():
            counts_by_alpha[key] = np.array([])
            for p in self.foi_dist:
                counts_by_alpha[key] = np.append(counts_by_alpha[key], p * val)
            counts_by_alpha[key] = counts_by_alpha[key].astype(int)
        self._init_counts = counts_by_alpha

    def import_race_data(self):
        """
        Parameters: None
        Returns:    None
        """

        # We first set the proportions for each race in the US.
        all_races = ['white', 'african_american', 'hispanic']
        all_races += ['native_american', 'asian', 'hawaiian']
        all_props = [0.637, 0.133, 0.158, 0.013, 0.057, 0.002]
        race_to_props = dict(zip(all_races, all_props))
        # Import the data from the race file.
        with open(self.gen_fp('races.csv'), 'r') as f:
            reader = csv.reader(f)
            line = ''
            while line != 'races:':
                line = next(reader)
                if len(line) > 0:
                    line = line[0]
            races = []
            for row in reader:
                if len(row) > 0:
                    races += [row[0]]
                else:
                    break
        race_props = [race_to_props[race] for race in races]
        # If not all races are selected, append 'other' to the list.
        if sorted(races) != sorted(all_races):
            races += ['other']
            race_props += [1.0 - sum(race_props)]
        self._races = races
        self._race_props = race_props
        self._nraces = len(races)

    def import_rate_data(self):
        """
        Parameters: None
        Returns:    None
        """

        # In what follows, the standard parameter files contain the mean
        # value lists, while the stochastic contain two lists, the first for
        # the reversion parameter, and the second for the diffustion. 
        # Import initiation rates.
        with open(self.gen_fp('initiation.json'), 'r') as f:
            init_data = json.load(f)
        # Convert keys from strings to tuples.
        lambdas = {}
        for key_0, val_0 in init_data.items():
            key_0 = ast.literal_eval(key_0)
            lambdas[key_0] = {}
            for key_1, val_1 in val_0.items():
                lambdas[key_0][ast.literal_eval(key_1)] = np.array(val_1)
        self._lambda = lambdas
        # The stochastic component for lambda.
        if self.scheck:
            with open(self.gen_fp('initiation_stochastic.json'), 'r') as f:
                init_data = json.load(f)
            # Convert keys from strings to tuples.
            s_lambda = {}
            for key_0, val_0 in init_data.items():
                key_0 = ast.literal_eval(key_0)
                s_lambda[key_0] = {}
                for key_1, val_1 in val_0.items():
                    key_1 = ast.literal_eval(key_1)
                    # We will append the mean value from self.lambdas.
                    u_0 = np.array([self.lambdas[key_0][key_1]]).T
                    u_1 = np.array(val_1)
                    s_lambda[key_0][key_1] = np.append(u_0, u_1, axis=1)
            self._s_lambda = s_lambda
        # The relapse rates.
        with open(self.gen_fp('relapse.json'), 'r') as f:
            relapse_data = json.load(f)
        # Convert keys from strings to tuples.
        gamma = {}
        for key_0, val_0 in relapse_data.items():
            key_0 = ast.literal_eval(key_0)
            gamma[key_0] = {}
            for key_1, val_1 in val_0.items():
                gamma[key_0][ast.literal_eval(key_1)] = np.array(val_1)
        self._gamma = gamma
        # The stochastic component for gamma.
        if self.scheck:
            with open(self.gen_fp('relapse_stochastic.json'), 'r') as f:
                relapse_data = json.load(f)
            # Convert keys from strings to tuples.
            s_gamma = {}
            for key_0, val_0 in relapse_data.items():
                key_0 = ast.literal_eval(key_0)
                s_gamma[key_0] = {}
                for key_1, val_1 in val_0.items():
                    key_1 = ast.literal_eval(key_1)
                    # We will append the mean value from self.gamma.
                    u_0 = np.array([self.gamma[key_0][key_1]]).T
                    u_1 = np.array(val_1)
                    s_gamma[key_0][key_1] = np.append(u_0, u_1, axis=1)
            self._s_gamma = s_gamma
        # The cessation rates.
        with open(self.gen_fp('cessation.json'), 'r') as f:
            cessation_data = json.load(f)
        # Convert keys from strings to tuples.
        chi = {}
        for key_0, val_0 in cessation_data.items():
            key_0 = ast.literal_eval(key_0)
            chi[key_0] = {}
            for key_1, val_1 in val_0.items():
                chi[key_0][ast.literal_eval(key_1)] = np.array(val_1)
        self._chi = chi
        # The stochastic component for chi.
        if self.scheck:
            with open(self.gen_fp('cessation_stochastic.json'), 'r') as f:
                cessation_data = json.load(f)
            # Convert keys from strings to tuples.
            s_chi = {}
            for key_0, val_0 in cessation_data.items():
                key_0 = ast.literal_eval(key_0)
                s_chi[key_0] = {}
                for key_1, val_1 in val_0.items():
                    key_1 = ast.literal_eval(key_1)
                    # We will append the mean value from self.chi.
                    u_0 = np.array([self.chi[key_0][key_1]]).T
                    u_1 = np.array(val_1)
                    s_chi[key_0][key_1] = np.append(u_0, u_1, axis=1)
            self._s_chi = s_chi
        # The health transitions.
        with open(self.gen_fp('health.json'), 'r') as f:
            health_data = json.load(f)
        # Convert keys from strings to tuples.
        kappa = {}
        for key_0, val_0 in health_data.items():
            key_0 = ast.literal_eval(key_0)
            kappa[key_0] = {}
            for key_1, val_1 in val_0.items():
                kappa[key_0][ast.literal_eval(key_1)] = np.array(val_1)
        self._kappa = kappa
        # The stochastic component for kappa.
        if self.scheck:
            with open(self.gen_fp('health_stochastic.json'), 'r') as f:
                health_data = json.load(f)
            # Convert keys from strings to tuples.
            s_kappa = {}
            for key_0, val_0 in health_data.items():
                key_0 = ast.literal_eval(key_0)
                s_kappa[key_0] = {}
                for key_1, val_1 in val_0.items():
                    key_1 = ast.literal_eval(key_1)
                    # We will append the mean value from self.kappa.
                    u_0 = np.array([self.kappa[key_0][key_1]]).T
                    u_1 = np.array(val_1)
                    s_kappa[key_0][key_1] = np.append(u_0, u_1, axis=1)
            self._s_kappa = s_kappa
        # The deaths.
        with open(self.gen_fp('death.json'), 'r') as f:
            death_data = json.load(f)
        # Convert keys from strings to tuples.
        death = {}
        for key_0, val_0 in death_data.items():
            key_0 = ast.literal_eval(key_0)
            death[key_0] = {}
            for key_1, val_1 in val_0.items():
                death[key_0][ast.literal_eval(key_1)] = np.array(val_1)
        self._death = death
        # Import addiction increase rates.
        with open(self.gen_fp('addiction_up.json'), 'r') as f:
            addiction_up_data = json.load(f)
        # Convert keys from strings to ints.
        phi = {}
        for key, val in addiction_up_data.items():
            phi[int(key)] = val
        self._phi = phi
        # Import addiction decrease rates.
        with open(self.gen_fp('addiction_down.json'), 'r') as f:
            addiction_down_data = json.load(f)
        # Convert keys from strings to ints.
        psi = {}
        for key, val in addiction_down_data.items():
            psi[int(key)] = val
        self._psi = psi

    def import_stock_counts(self):
        """
        Parameters: None
        Returns:    w: int
                        The count of stocks for each (sex, race) pair.
        """

        # Import the information from the file.
        with open('{0}/stock_counts.csv'.format(self.data_dir), 'r') as f:
            reader = csv.reader(f)
            line = ''
            # Loop until we arrive at the stock counts.
            while line != 'stock_counts:':
                line = next(reader)
                if len(line) > 0:
                    line = line[0]
            # Loop the remaining row and assign values.
            self.sizes = {}
            for row in reader:
                if len(row) > 1:
                    self.sizes[row[0]] = int(row[1])
        # Having now the counts, we determine the final size.
        t_size = self.sizes['tobacco_states']
        t_size **= self.sizes['tobacco_products']
        h_size = self.sizes['health_states']
        h_size **= self.sizes['health_conditions']
        a_size = self.sizes['addiction_levels']
        max_size = t_size * h_size * a_size * self.sizes['age_groups']
        # The true size is the maximum size, minus the stocks that cannot
        # exist, namely those that have levels of addiction for individuals who
        # have never used tobacco.
        null_size = h_size * (a_size - 1) * self.sizes['age_groups']
        w = max_size - null_size
        return w

    def import_theta(self):
        """
        Parameters: None
        Returns:    None
        """

        # Check if the file exists.
        fp = self.gen_fp('theta.json')
        if os.path.isfile(fp):
            # Import the data.
            with open(fp, 'r') as f:
                data = json.load(f)
            theta = {}
            for key, val in data.items():
                theta[ast.literal_eval(key)] = val
            self._theta = theta
        else:
            self._theta = None

    def product(self, stock):
        """
        Parameters: stock: str or list
                        If str, then it is a stock string, chosen from 'S',
                        'I', 'U', or 'Z'. If a list, then it is a list of
                        integers that define respective ranges.
        Returns:    w: itertools.product
                        The product for the ranges associated to stock.
        """

        if isinstance(stock, str):
            if stock == 'S':
                # Susceptible has only three subdivisions, determined by age.
                return range(self.sizes['age_groups'])
            if stock == 'I':
                # The range for (health_1, ..., health_k, age).
                ranges = ()
                num_health = self.sizes['health_conditions']
                for x in range(num_health):
                    ranges += (self.sizes['health_states'], )
                ranges += (self.sizes['age_groups'], )
                rng = itertools.product(*[range(x) for x in ranges])
                # We exclude any cases of (0, 0, v), for all v.
                return (x for x in rng if sum(x[:num_health]) != 0)
            if stock == 'U':
                # The range for (use_1, ..., use_n, addiction, age).
                ranges = ()
                num_prod = self.sizes['tobacco_products']
                for x in range(num_prod):
                    ranges += (self.sizes['tobacco_states'], )
                ranges += (self.sizes['addiction_levels'], )
                ranges += (self.sizes['age_groups'], )
                rng = itertools.product(*[range(x) for x in ranges])
                # We exclude any cases of (0, 0, a, v).
                return (x for x in rng if sum(x[:num_prod]) != 0)
            if stock == 'Z':
                # The ranges for (use_1, ..., use_n, health_1, ..., health_k,
                # addiction, age).
                ranges = ()
                num_health = self.sizes['health_conditions']
                num_prod = self.sizes['tobacco_products']
                for x in range(num_prod):
                    ranges += (self.sizes['tobacco_states'], )
                for x in range(num_health):
                    ranges += (self.sizes['health_states'], )
                ranges += (self.sizes['addiction_levels'], )
                ranges += (self.sizes['age_groups'], )
                rng = itertools.product(*[range(x) for x in ranges])
                # We exclude the cases of (0, 0, z, w, a, v) and (x, y, 0, 0,
                # a, v).
                t0 = num_prod
                t1 = t0 + num_health
                return (x for x in rng if sum(x[:t0]) * sum(x[t0:t1]) != 0)
        else:
            return itertools.product(*[range(x) for x in stock])

    def run(self):
        """
        Parameters: None
        Returns:    None
        """

        # Run the deterministic model simulations. We set a MFSG instance for
        # for each (sex, race) pair.
        for x, key in enumerate(self.product((2, self.nraces))):
            self.run_sex_race(x, key)

    def run_sex_race(self, x, key):
        """
        Parameters: x: int
                        The enumerated index of the sex, race key.
                    key: tuple
                        The length two tuple denoting (sex, race).
        Returns:    None
        """

        # The MFSG ID (self.sruns == 1 if self.scheck == 0).
        m = x / (2 * self.nraces * self.sruns)
        self.mfsg[key] = MeanFieldSolutionGenerator(self, m, key)
        # If no theta file exists, then generate one now.
        self.gen_theta_file()
        # Convert the densities to counts. This will set self._counts.
        self.gen_counts()
        # Get the death counts. This will set self._death_counts.
        self.gen_death_counts()
        # Output to file.
        self.write()

    def set_output_columns(self):
        """
        Parameters: None
        Returns:    None
        """

        cols = ['Initial']
        # For each year we will have output values for each timestep, as well
        # as the initial values.
        col_range = range(0, self.years * self.steps + 1)
        # Set the column labels based on the number of steps.
        if self.steps == 365:
            # In this case, each step represents one day.
            cols += ['Day {0}'.format(x) for x in col_range]
        elif self.steps == 12:
            # In this case, each step represents one month.
            cols += ['Month {0}'.format(x) for x in col_range]
        elif self.steps == 1:
            # In this case, each step represents one year.
            cols += ['Year {0}'.format(x) for x in col_range]
        else:
            # In this case, we have some arbitrary steps.
            total_steps = self.years * self.steps + 1
            cols += np.linspace(0, self.years, total_steps).tolist()
        self._columns = cols

    def set_demographics(self):
        """
        Parameters: None
        Returns:    None
        """

        # The proportion of males and females in the US.
        sex = [0.4924, 0.5076]
        # Import the race data and subdivide population by (sex, race) pairs.
        # This will set self._races, self._race_props, and self._nraces.
        self.import_race_data()
        us_pop = {}
        for key in self.product((2, self.nraces)):
            us_pop[key] = sex[key[0]] * self.race_props[key[1]]
        self._us_pop = us_pop
        # Set the age transitions. These can be calibrated as needed.
        self._age = {0:0.0348, 1:0.0235}

    def stock_index(self, key, alpha=1):
        """
        Parameters: key: tuple
                        The stock key of the form (x, y, z, w, a, v). Note that
                        in the event of stocks where the full tuple is not
                        defined, such as U stocks with have keys of the form
                        (x, y, a, v), we extend by setting the missing values
                        to 0, for example, (x, y, a, v) --> (x, y, 0, 0, a, v).
                    alpha: int
                        The force of influence associated to the given stock.
        Returns:    idx: int
                        The index of the given stock in the list of all stocks
                        defined by
                            [S_1, I_1, U_1, Z_1, ..., S_n, I_n, U_n, Z_n],
                        where for each i in [1, ..., n], and stock X in [S, I,
                        U, Z], X_i is the list of 'X' stocks for alpha = i.
        """

        return self.canonical_list.index(key) + (alpha - 1) * self.size

    def update_rates(self, dt):
        """
        Parameters: dt: float
                        The step size in the discretized CIR.
        Returns:    None
        """

        # Each parameter dictionary has the same first set of keys, so we use
        # lambdas to define the loop.
        for sr in self.lambdas.keys():
            # Update the lambda values.
            for key, val in self.s_lambda[sr].items():
                cir = []
                for x in range(len(val)):
                    # The CIR parameters.
                    b, a, c = val[x]
                    # The most recent CIR value.
                    r = self.lambdas[sr][key][x]
                    # The new CIR value.
                    r_new = -r
                    while r_new + r <= 0 or r_new + r >= 1:
                        # The random component.
                        e = np.random.normal(0, 1)
                        r_new = a * (b - r) * dt + c * np.sqrt(r * dt) * e
                    r += r_new
                    cir += [r]
                self._lambda[sr][key] = np.array(cir)
            # Update the gamma values.
            for key, val in self.s_gamma[sr].items():
                cir = []
                for x in range(len(val)):
                    # The CIR parameters.
                    b, a, c = val[x]
                    # The most recent CIR value.
                    r = self.gamma[sr][key][x]
                    # The new CIR value.
                    r_new = -r
                    while r_new + r <= 0 or r_new + r >= 1:
                        # The random component.
                        e = np.random.normal(0, 1)
                        r_new = a * (b - r) * dt + c * np.sqrt(r * dt) * e
                    r += r_new
                    cir += [r]
                self._gamma[sr][key] = np.array(cir)
            # Update the chi values.
            for key, val in self.s_chi[sr].items():
                cir = []
                for x in range(len(val)):
                    # The CIR parameters.
                    b, a, c = val[x]
                    # The most recent CIR value.
                    r = self.chi[sr][key][x]
                    # The new CIR value.
                    r_new = -r
                    while r_new + r <= 0 or r_new + r >= 1:
                        # The random component.
                        e = np.random.normal(0, 1)
                        r_new = a * (b - r) * dt + c * np.sqrt(r * dt) * e
                    r += r_new
                    cir += [r]
                self._chi[sr][key] = np.array(cir)
            # Update the kappa values.
            for key, val in self.s_kappa[sr].items():
                cir = []
                for x in range(len(val)):
                    # The CIR parameters.
                    b, a, c = val[x]
                    # The most recent CIR value.
                    r = self.kappa[sr][key][x]
                    # The new CIR value.
                    r_new = -r
                    while r_new + r <= 0 or r_new + r >= 1:
                        # The random component.
                        e = np.random.normal(0, 1)
                        r_new = a * (b - r) * dt + c * np.sqrt(r * dt) * e
                    r += r_new
                    cir += [r]
                self._kappa[sr][key] = np.array(cir)
        # Now generate the updated stock rates.
        self.gen_stock_rates()

    def write(self):
        """
        Parameters: None
        Returns:    None
        """

        # Set 'deterministic' or 'stochastic' fid.
        if self.scheck:
            fid = 'stochastic'
        else:
            fid = 'deterministic'
        # We need an output folder if it does not already exist.
        if not os.path.exists('output'):
            os.makedirs('output')
        for key in self.mfsg:
            if key[0] == 0:
                s = 'male'
            else:
                s = 'female'
            r = self.races[key[1]]
            if self.scheck:
                tup_in = (fid, self.sruns, s, r, self.years, self.steps)
                fpd = 'output/densities_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpc = 'output/counts_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpdc = 'output/death_counts_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpdl = 'output/densities_low_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpcl = 'output/counts_low_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpdcl = 'output/death_counts_low_'
                fpdcl += '%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpdh = 'output/densities_high_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpch = 'output/counts_high_%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
                fpdch = 'output/death_counts_high_'
                fpdch += '%s_n%i_%s_%s_y%i_s%i.csv' % tup_in
            else:
                tup_in = (fid, s, r, self.years, self.steps)
                fpd = 'output/densities_%s_%s_%s_y%i_s%i.csv' % tup_in
                fpc = 'output/counts_%s_%s_%s_y%i_s%i.csv' % tup_in
                fpdc = 'output/death_counts_%s_%s_%s_y%i_s%i.csv' % tup_in
            with open(fpd, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([''] + list(self.columns))
                for x, row in enumerate(self.mfsg[key].sol):
                    new_row = [self.rows[x], self.y0[key][x]] + list(row)
                    writer.writerow(new_row)
            with open(fpc, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([''] + list(self.columns))
                for x, row in enumerate(self.counts[key]):
                    new_row = [self.rows[x], self.init_counts[key][x]]
                    new_row += list(row)
                    writer.writerow(new_row)
            with open(fpdc, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([''] + list(self.columns)[1:])
                for x, row in enumerate(self.death_counts[key]):
                    writer.writerow([self.rows[x]] + list(row))
            if self.scheck:
                with open(fpdl, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns))
                    for x, row in enumerate(self.mfsg[key].sol_low):
                        new_row = [self.rows[x], self.y0[key][x]] + list(row)
                        writer.writerow(new_row)
                with open(fpcl, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns))
                    for x, row in enumerate(self.counts_low[key]):
                        new_row = [self.rows[x], self.init_counts[key][x]]
                        new_row += list(row)
                        writer.writerow(new_row)
                with open(fpdcl, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns)[1:])
                    for x, row in enumerate(self.death_counts_low[key]):
                        writer.writerow([self.rows[x]] + list(row))
                with open(fpdh, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns))
                    for x, row in enumerate(self.mfsg[key].sol_high):
                        new_row = [self.rows[x], self.y0[key][x]] + list(row)
                        writer.writerow(new_row)
                with open(fpch, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns))
                    for x, row in enumerate(self.counts_high[key]):
                        new_row = [self.rows[x], self.init_counts[key][x]]
                        new_row += list(row)
                        writer.writerow(new_row)
                with open(fpdch, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([''] + list(self.columns)[1:])
                    for x, row in enumerate(self.death_counts_high[key]):
                        writer.writerow([self.rows[x]] + list(row))

    # The read-only class properties.
    @property
    def age(self):
        return self._age

    @property
    def canonical_list(self):
        return self._canonical_list

    @property
    def chi(self):
        return self._chi

    @property
    def columns(self):
        return self._columns

    @property
    def counts(self):
        return self._counts

    @property
    def counts_high(self):
        return self._counts_high

    @property
    def counts_low(self):
        return self._counts_low

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def death(self):
        return self._death

    @property
    def death_counts(self):
        return self._death_counts

    @property
    def death_counts_low(self):
        return self._death_counts_low

    @property
    def death_counts_high(self):
        return self._death_counts_high

    @property
    def dy_death(self):
        return self._dy_death

    @property
    def exp_foi(self):
        return self._exp_foi

    @property
    def foi_range(self):
        return self._foi_range

    @property
    def gamma(self):
        return self._gamma

    @property
    def init_counts(self):
        return self._init_counts

    @property
    def kappa(self):
        return self._kappa

    @property
    def lambdas(self):
        return self._lambda

    @property
    def nraces(self):
        return self._nraces

    @property
    def num_foi(self):
        return self._num_foi

    @property
    def phi(self):
        return self._phi

    @property
    def psi(self):
        return self._psi

    @property
    def race_props(self):
        return self._race_props

    @property
    def races(self):
        return self._races

    @property
    def rates(self):
        return self._rates

    @property
    def rows(self):
        return self._rows

    @property
    def size(self):
        return self._size

    @property
    def s_chi(self):
        return self._s_chi

    @property
    def s_gamma(self):
        return self._s_gamma

    @property
    def s_kappa(self):
        return self._s_kappa

    @property
    def s_lambda(self):
        return self._s_lambda

    @property
    def stock_size(self):
        return self._stock_size

    @property
    def t(self):
        return self._t

    @property
    def theta(self):
        return self._theta

    @property
    def theta_indices(self):
        return self._theta_indices

    @property
    def us_pop(self):
        return self._us_pop

    @property
    def us_pop_total(self):
        return self._us_pop_total

    @property
    def y0(self):
        return self._y0

    # Read-write class properties.
    @property
    def foi_dist(self):
        return self._foi_dist

    @foi_dist.setter
    def foi_dist(self, val):
        check_pos = all([x >= 0 for x in val])
        if len(val) == self.num_foi and sum(val) == 1 and check_pos:
            self._foi_dist = val
        else:
            raise ValueError('FOI distribution must be positive and sum to 1.')

    @property
    def foi_param(self):
        return self._foi_param

    @foi_param.setter
    def foi_param(self, val):
        if 0 < val <= 1:
            self._foi_param = val
        else:
            raise ValueError('Network parameter must be value in (0, 1].')

    @property
    def gui(self):
        return self._gui

    @gui.setter
    def gui(self, val):
        if val == 0:
            self._gui = None
        else:
            self._gui = val

    @property
    def init_fp(self):
        return self._init_fp

    @init_fp.setter
    def init_fp(self, val):
        if os.path.isfile('{0}/{1}'.format(self.data_dir, val)):
            return '{0}/{1}'.format(self.data_dir, val)
        else:
            raise ValueError('Initial count filepath does not exist.')

    @property
    def scheck(self):
        return self._scheck

    @scheck.setter
    def scheck(self, val):
        if val in [True, False, 0, 1]:
            return bool(val)
        else:
            raise ValueError('Stochastic check must be True/False or 1/0.')

    @property
    def sruns(self):
        return self._sruns

    @sruns.setter
    def sruns(self, val):
        if self.scheck and int(val) > 0:
            return int(val)
        elif self.scheck:
            raise ValueError('The number of stochastic runs must be positive.')
        else:
            self._scheck = val

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, val):
        if int(val) > 0:
            self._steps = int(val)
        else:
            raise ValueError('Steps must be a positive integer.')
    @property
    def years(self):
        return self._years

    @years.setter
    def years(self, val):
        if val > 0:
            self._years = val
        else:
            raise ValueError('Years must be a positive value.')


class MeanFieldSolutionGenerator():
    """
    Class used to solve the mean-field equations.
    """

    def __init__(self, sdm, m_id, sr):
        """
        Parameters: sdm: SystemDynamicsManager()
                        The parent SDM.
                    m_id: float
                        The position of the class instance in the set of all
                        instances, divided by the count of instances. This is
                        only used to let the progress bar know where to start.
                    sr: tuple
                        The sex, race pair for the MFSG.
        Returns:    None
        """

        # Set the sdm and id properties.
        self._sdm = sdm
        self._id = m_id
        self._sex_race = sr
        # The current time value is used for the progress bar.
        self._current_t = 0
        # Set the transition rates.
        self.rates = sdm.rates[sr]
        # Set the max time over which the system is solved.
        self.t_max = max(sdm.t)
        # The following attribute is used to ensure the GUI progress bar does
        # not decrease.
        self.t_step = m_id
        # We first roughly estimate the steady state of the system. This will
        # generate the property self._theta.
        self.gen_theta()
        # Given the steady state, we can solve the system. This will set the
        # property self._sol.
        print('Now solving for the sex-race pair {0}.'.format(sr))
        self.solve()

    def dy_foi(self, t, f, alpha):
        """
        Parameters: t: float
                        The time parameter (not needed in the deterministic
                        case).
                    f: list
                        The list of stock counts, ordered according to:
                        [S, I, U, Z], where for each stock X in [S, I, U, Z],
                        X is the list of 'X' stocks. For example, with three
                        age groups, S = S(0), S(1), S(2).
                    alpha:
                        The force of influence associated to this system.
        Returns:    dydt: list
                        The list of derivatives of y for alpha stocks.
        """

        self.current_t = t
        return self.dy_steady_foi(f, self.theta, alpha)

    def dy_steady(self, f):
        """
        Parameters: f: list
                        The list of stock counts, ordered according to:
                        [S_1, I_1, U_1, Z_1, ..., S_n, I_n, U_n, Z_n], where
                        for each i in [1, ..., n], and stock X in [S, I, U, Z],
                        X_i is the list of 'X' stocks for alpha = i. For
                        example, with three age groups, S_1 = S_1(0), S_1(1),
                        S_1(2).
        Returns:    dydt: list
                        The list of derivatives of f.
        """

        # Set theta for the current f input, and allocate counters for
        # successive theta values.
        t = self.theta_function(f)
        dydt = []
        # Given the value of theta, the dydt values for each alpha can be
        # determined independently.
        for alpha in self.sdm.foi_range:
            # Get the list index bounds for alpha.
            lower = (alpha - 1) * self.sdm.size
            upper = alpha * self.sdm.size
            dydt += self.dy_steady_foi(f[lower:upper], t, alpha)
        return dydt

    def dy_steady_foi(self, f, t, alpha):
        """
        Parameters: f: list
                        The list of stock counts, ordered according to:
                        [S, I, U, Z], where for each stock X in [S, I, U, Z],
                        X is the list of 'X' stocks. For example, with three
                        age groups, S = S(0), S(1), S(2).
                    t: list
                        The list of force of influce mean effects.
                    alpha:
                        The force of influence associated to this system.
        Returns:    dydt: list
                        The list of derivatives of y for alpha stocks.
        """

        dydt = []
        # Set the values for each stock, starting with the S stock densities.
        for key in self.sdm.product('S'):
            v = key
            s_idx = self.sdm.stock_index((0, 0, 0, 0, 0, v))
            f_in = 0
            f_out = 0
            # The inflow for S is the birth rate when v == 0, and is the
            # age outflow of S[v - 1] otherwise.
            if v == 0:
                # To ensure system stability, we assume that the birth rate
                # is equal to the total current death rate.
                f_in += np.dot(f, self.sdm.dy_death[self.sex_race])
            else:
                idx_0 = self.sdm.stock_index((0, 0, 0, 0, 0, v - 1))
                f_in += self.rates['S']['age'][v - 1] * f[idx_0]
            # The outflow due to initiation and illness
            f_out += alpha * np.dot(self.rates['S']['lambda'][v], t)
            f_out += sum(self.rates['S']['kappa'][v])
            # The outflow due to death.
            f_out += self.sdm.death[self.sex_race][(0, 0, 0, 0, 0, v)]
            # The outflow due to aging.
            if v < 2:
                f_out += self.rates['S']['age'][v]
            dydt += list(f_in - f_out * f[s_idx])
        # Loop through the I stocks,
        for key in self.sdm.product('I'):
            z, w, v = key
            i_idx = self.sdm.stock_index((0, 0, z, w, 0, v))
            f_in = 0
            f_out = 0
            # The inflow.
            for idx in range(2):
                if key[idx] == 0:
                    # Inflow from susceptible stock.
                    idx_0 = self.sdm.stock_index((0, 0, 0, 0, 0, v))
                    f_in += self.rates['S']['kappa'][v][1 - idx] * f[idx_0]
            if z == w == 1:
                # Inflow from each illness set.
                idx_0 = self.sdm.stock_index((0, 0, 0, 1, 0, v))
                f_in += self.rates['I']['kappa'][(0, 1, v)] * f[idx_0]
                idx_0 = self.sdm.stock_index((0, 0, 1, 0, 0, v))
                f_in += self.rates['I']['kappa'][(1, 0, v)] * f[idx_0]
            # Inflow due to aging.
            if v > 0:
                idx_0 = self.sdm.stock_index((0, 0, z, w, 0, v - 1))
                f_in += self.rates['I']['age'][(z, w, v - 1)] * f[idx_0]
            # Outflow due to developing other illness.
            if z == 0 or w == 0:
                f_out += self.rates['I']['kappa'][key]
            # Outflow due to product initiation.
            f_out += alpha * np.dot(self.rates['I']['lambda'][key], t)
            # Outflow due to death.
            f_out += self.sdm.death[self.sex_race][(0, 0, z, w, 0, v)]
            # Outflow due to aging.
            if v < 2:
                f_out += self.rates['I']['age'][key]
            dydt += list(f_in - f_out * f[i_idx])
        # Loop through the U stocks.
        for key in self.sdm.product('U'):
            x, y, a, v = key
            u_idx = self.sdm.stock_index((x, y, 0, 0, a, v))
            f_in = 0
            f_out = 0
            # Inflow due to product initiation of never users.
            idx_0 = self.sdm.stock_index((0, 0, 0, 0, 0, v))
            if (x, y, a) == (1, 0, 0):
                f_in_0 = alpha * self.rates['S']['lambda'][v][0] * t[0]
                f_in += f_in_0 * f[idx_0]
            if (x, y, a) == (0, 1, 0):
                f_in_0 = alpha * self.rates['S']['lambda'][v][1] * t[1]
                f_in +=  f_in_0 * f[idx_0]
            # Inflow due to relapse.
            if x == 1:
                idx_0 = self.sdm.stock_index((2, y, 0, 0, a, v))
                if y == 2:
                    f_in_0 = self.rates['U']['gamma'][(2, y, a, v)][0] * t[0]
                else:
                    f_in_0 = self.rates['U']['gamma'][(2, y, a, v)] * t[0]
                f_in += f_in_0 * alpha * f[idx_0]
            if y == 1:
                idx_0 = self.sdm.stock_index((x, 2, 0, 0, a, v))
                if x == 2:
                    f_in_0 = self.rates['U']['gamma'][(x, 2, a, v)][1] * t[1]
                else:
                    f_in_0 = self.rates['U']['gamma'][(x, 2, a, v)] * t[1]
                f_in += f_in_0 * alpha * f[idx_0]
            # Inflow due to cessation of the other product.
            if x == 2:
                idx_0 = self.sdm.stock_index((1, y, 0, 0, a, v))
                if y == 1:
                    f_in_0 = self.rates['U']['chi'][(1, y, a, v)][0]
                else:
                    f_in_0 = self.rates['U']['chi'][(1, y, a, v)]
                f_in += f_in_0 * f[idx_0]
            if y == 2:
                idx_0 = self.sdm.stock_index((x, 1, 0, 0, a, v))
                if x == 1:
                    f_in_0 = self.rates['U']['chi'][(x, 1, a, v)][1]
                else:
                    f_in_0 = self.rates['U']['chi'][(x, 1, a, v)]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to initiation of product 1 by current and former
            # users of product 2.
            if x == 1 and y != 0:
                idx_0 = self.sdm.stock_index((0, y, 0, 0, a, v))
                f_in_0 = alpha * self.rates['U']['lambda'][(0, y, a, v)] * t[0]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to initiation of product 2 by current and former
            # users of product 1.
            if y == 1 and x != 0:
                idx_0 = self.sdm.stock_index((x, 0, 0, 0, a, v))
                f_in_0 = alpha * self.rates['U']['lambda'][(x, 0, a, v)] * t[1]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to increased addiction of current users.
            if (x == 1 or y == 1) and a > 0:
                idx_0 = self.sdm.stock_index((x, y, 0, 0, a - 1, v))
                f_in += self.rates['U']['phi'][(x, y, a - 1, v)] * f[idx_0]
            # Inflow due to decreased addiction of former users.
            if (x != 1 and y != 1) and a < 3:
                idx_0 = self.sdm.stock_index((x, y, 0, 0, a + 1, v))
                f_in += self.rates['U']['psi'][(x, y, a + 1, v)] * f[idx_0]
            # Inflow due to aging.
            if v > 0:
                idx_0 = self.sdm.stock_index((x, y, 0, 0, a, v - 1))
                f_in += self.rates['U']['age'][(x, y, a, v - 1)] * f[idx_0]
            # Outflow due to death.
            f_out += self.sdm.death[self.sex_race][(x, y, 0, 0, a, v)]
            # Outflow due to developing a health condition.
            f_out += sum(self.rates['U']['kappa'][key])
            # Outflow due to increased addiction of a currently used
            # product.
            if (x == 1 or y == 1) and a < 3:
                f_out += self.rates['U']['phi'][key]
            # Outflow due to decreased addiction of a fomerly used product.
            if (x != 1 and y != 1) and a > 0:
                f_out += self.rates['U']['psi'][(x, y, a, v)]
            # Outflow due to product initiation.
            if x == 0:
                f_out += alpha * self.rates['U']['lambda'][key] * t[0]
            if y == 0:
                f_out += alpha * self.rates['U']['lambda'][key] * t[1]
            # Outflow due to relapse.
            if (x, y) == (2, 2):
                f_out += alpha * np.dot(self.rates['U']['gamma'][key], t)
            elif x == 2:
                f_out += alpha * self.rates['U']['gamma'][key] * t[0]
            elif y == 2:
                f_out += alpha * self.rates['U']['gamma'][key] * t[1]
            # Outflow due to cessation.
            if (x, y) == (1, 1):
                f_out += sum(self.rates['U']['chi'][key])
            elif x == 1 or y == 1:
                f_out += self.rates['U']['chi'][key]
            # Outflow due to aging.
            if v < 2:
                f_out += self.rates['U']['age'][key]
            dydt += list(f_in - f_out * f[u_idx])
        # Loop through Z stocks.
        for key in self.sdm.product('Z'):
            x, y, z, w, a, v = key
            z_idx = self.sdm.stock_index(key)
            key_I = (z, w, v)
            key_U = (x, y, a, v)
            f_in = 0
            f_out = 0
            # Inflow due to product initiation from infected individuals.
            idx_0 = self.sdm.stock_index((0, 0, z, w, 0, v))
            if (x, y, a) == (1, 0, 0):
                f_in_0 = alpha * self.rates['I']['lambda'][key_I][0] * t[0]
                f_in += f_in_0 * f[idx_0]
            if (x, y, a) == (0, 1, 0):
                f_in_0 = alpha * self.rates['I']['lambda'][key_I][1] * t[1]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to condition development by current and former
            # users.
            if (z != 1) or (w != 1):
                idx_0 = self.sdm.stock_index((x, y, 0, 0, a, v))
                f_in += self.rates['U']['kappa'][key_U][w] * f[idx_0]
            # Inflow due to increased level of addiction from a currently
            # used product.
            if (x == 1 or y == 1) and a > 0:
                key_0 = (x, y, z, w, a - 1, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in += self.rates['Z']['phi'][key_0] * f[idx_0]
            # Inflow due to decreased level of addiction from a formerly
            # used product.
            if (x != 1 and y != 1) and a < 3:
                key_0 = (x, y, z, w, a + 1, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in += self.rates['Z']['psi'][key_0] * f[idx_0]
            # Inflow due to initiation of product 1 by current or former
            # users of product 2.
            if x == 1 and y != 0:
                key_0 = (0, y, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in_0 = alpha * self.rates['Z']['lambda'][key_0] * t[0]
                f_in +=  f_in_0 * f[idx_0]
            # Inflow due to initiation of product 2 by current or former
            # users of product 1.
            if x != 0 and y == 1:
                key_0 = (x, 0, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in_0 = alpha * self.rates['Z']['lambda'][key_0] * t[1]
                f_in +=  f_in_0 * f[idx_0]
            # Inflow due to development of a health condition.
            if (z, w) == (1, 1):
                key_0 = (x, y, 0, 1, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in += self.rates['Z']['kappa'][key_0] * f[idx_0]
                key_0 = (x, y, 0, 1, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                f_in += self.rates['Z']['kappa'][key_0] * f[idx_0]
            # Inflow due to relapse.
            if x == 1:
                key_0 = (2, y, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                if y == 2:
                    f_in_0 = alpha * self.rates['Z']['gamma'][key_0][0] * t[0]
                else:
                    f_in_0 = alpha * self.rates['Z']['gamma'][key_0] * t[0]
                f_in += f_in_0 * f[idx_0]
            if y == 1:
                key_0 = (x, 2, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                if x == 2:
                    f_in_0 = alpha * self.rates['Z']['gamma'][key_0][1] * t[1]
                else:
                    f_in_0 = alpha * self.rates['Z']['gamma'][key_0] * t[1]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to cessation.
            if x == 2:
                key_0 = (1, y, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                if y == 1:
                    f_in_0 = self.rates['Z']['chi'][key_0][0]
                else:
                    f_in_0 = self.rates['Z']['chi'][key_0]
                f_in += f_in_0 * f[idx_0]
            if y == 2:
                key_0 = (x, 1, z, w, a, v)
                idx_0 = self.sdm.stock_index(key_0)
                if x == 1:
                    f_in_0 = self.rates['Z']['chi'][key_0][1]
                else:
                    f_in_0 = self.rates['Z']['chi'][key_0]
                f_in += f_in_0 * f[idx_0]
            # Inflow due to aging.
            if v > 0:
                key_0 = (x, y, z, w, a, v - 1)
                idx_0 = self.sdm.stock_index(key_0)
                f_in += self.rates['Z']['age'][key_0] * f[idx_0]
            # Outflow due to death.
            f_out += self.sdm.death[self.sex_race][key]
            # Outflow due to increased addiction of current product.
            if (x == 1 or y == 1) and a < 3:
                f_out += self.rates['Z']['phi'][key]
            # Outflow due to decreased addiction of former product.
            if (x != 1 and y != 1) and a > 0:
                f_out_0 = self.rates['Z']['psi'][(x, y, z, w, a, v)]
            # Outflow due to health transition.
            if z == 0 or w == 0:
                f_out += self.rates['Z']['kappa'][key]
            # Outflow due to product initiation.
            if x == 0:
                f_out += alpha * self.rates['Z']['lambda'][key] * t[0]
            if y == 0:
                f_out += alpha * self.rates['Z']['lambda'][key] * t[1]
            # Outflow due to relapse.
            if (x, y) == (2, 2):
                for idx in range(2):
                    f_out_0 = alpha * self.rates['Z']['gamma'][key][idx]
                    f_out += f_out_0 * t[idx]
            elif x == 2:
                f_out += alpha * self.rates['Z']['gamma'][key] * t[0]
            elif y == 2:
                f_out += alpha * self.rates['Z']['gamma'][key] * t[1]
            # Outflow due to cessation.
            if (x, y) == (1, 1):
                for idx in range(2):
                    f_out += self.rates['Z']['chi'][key][idx]
            elif x == 1 or y == 1:
                f_out += self.rates['Z']['chi'][key]
            # Outflow due to aging.
            if v < 2:
                f_out += self.rates['Z']['age'][key]
            dydt += list(f_in - f_out * f[z_idx])
        return dydt

    def gen_theta(self, tol=1e-6, n_max=1000):
        """
        Parameters: tol: float
                        The tolerance allowed for the estimate.
                    n_max: int
                        The number of iterations to attempt before quitting, if
                        the tolerance is not met.
        Returns:    None
        """

        # We check if theta was imported from file.
        if self.sdm.theta is not None:
            self._theta = self.sdm.theta[self.sex_race]
        else:
            # We estimate the theta parameters by the steady state values.
            # Finding the steady state requires finding the root of the
            # differential system (i.e. when the derivative is zero). Since the
            # system can have multiple roots, we project year by year,
            # normalizing as needed, stopping when tolerance has been achieved.
            y0 = self.sdm.y0[self.sex_race]
            t = self.theta_function(y0)
            tol_check = False
            s = self.sdm.size
            msg = 'Now generating theta parameter for {0}. This may take '
            msg += 'awhile.'
            print(msg.format(self.sex_race))
            for _ in range(n_max):
                y0 += self.dy_steady(y0)
                # Since we project one year into the future, there is the chance
                # that we project into negative range, which we take care of now.
                y0 *= (y0 >= 0)
                for x in self.sdm.foi_range:
                    # The sub array of y0 associated to this foi.
                    x_arr = y0[(x - 1) * s: x * s]
                    y0[(x - 1) * s: x * s] = x_arr / np.sum(x_arr)
                # Get the new version of theta.
                t_new = self.theta_function(y0)
                if np.linalg.norm(np.array(t_new) - np.array(t)) < tol:
                    # In this case tolerance has been met.
                    t = t_new
                    tol_check = True
                    break
                t = t_new
            if not tol_check:
                print('Theta set, but tolerance was not met.')
            # Set self._theta.
            self._theta = t_new

    def refine_solution(self, sol_all):
        """
        Parameters: sol_all: array-like
                        The current solution or collection of stochastic
                        solutions.
        Returns:    sol: array
                        The mean solution.
                    sol_low: array
                        The lower 95% confidence band.
                    sol_high: array
                        The upper 95% confidence band.
        """

        # If not stochastic, we have nothing to do.
        if not self.sdm.scheck:
            return sol_all, None, None
        # In this case we are stochastic, and the solution is the mean over all
        # solutions (over the first axis).
        sol = np.mean(sol_all, axis=0)
        sol_high = np.percentile(sol_all, 97.5, axis=0)
        sol_low = np.percentile(sol_all, 2.5, axis=0)
        return sol, sol_low, sol_high

    def set_progress(self, x):
        """
        Parameters: x: float
                        A value in [0, 1].
        Returns:    None
        """

        self.sdm.gui.progress_var.set(x)
        self.sdm.gui.root.update_idletasks()

    def solve(self):
        """
        Parameters: None
        Returns:    None
        """

        for x, alpha in enumerate(self.sdm.foi_range):
            y_alpha = self.solve_step(x, alpha)
            if x == 0:
                sol_0 = y_alpha
            else:
                sol_0 += y_alpha
        sol, sol_low, sol_high = self.refine_solution(sol_0)
        # Normalize the solution and multiply by the race/sex proportion.
        sol *= self.sdm.us_pop[self.sex_race] / np.sum(sol, axis=0)
        if self.sdm.scheck:
            sol_low *= self.sdm.us_pop[self.sex_race] / np.sum(sol_low, axis=0)
            high_sum = np.sum(sol_high, axis=0)
            sol_high *= self.sdm.us_pop[self.sex_race] / high_sum
        # Set the class attribute.
        self._sol = sol
        self._sol_low = sol_low
        self._sol_high = sol_high

    def solve_foi(self, alpha):
        """
        Parameters: alpha: int
                        The force of influence associated to this system.
        Returns:    y: list
                        The list of y values for alpha stocks at each time.
        """

        # Get the list index bounds for alpha.
        lower = (alpha - 1) * self.sdm.size
        upper = alpha * self.sdm.size
        # The initial states.
        y0 = self.sdm.y0[self.sex_race][lower:upper]
        # Solve the mean field differential system for alpha stocks.
        t = [self.sdm.t[0], self.sdm.t[-1]]
        func = lambda t, f: self.dy_foi(t, f, alpha)
        return scipy.integrate.solve_ivp(func, t, y0, t_eval=self.sdm.t).y

    def solve_sfoi(self, alpha):
        """
        Parameters: alpha: int
                        The force of influence associated to this system.
        Returns:    y: list
                        The list of y values for alpha stocks at each time.
        """

        y = []
        # Get the list index bounds for alpha.
        lower = (alpha - 1) * self.sdm.size
        upper = alpha * self.sdm.size
        # The initial states.
        y0 = self.sdm.y0[self.sex_race][lower:upper]
        y += [y0]
        # Apply Euclidean method to estimate the output values, adjusting the
        # stochastic rates with each step.
        dt = self.sdm.years / self.sdm.steps
        for t in np.arange(0, self.sdm.t[-1] + dt, dt):
            # Update the rates.
            self.sdm.update_rates(dt)
            # Given the new rates, update y0.
            y0 += np.array(self.dy_foi(t, y0, alpha)) * dt
            y += [y0]
        return np.array(y).T

    def solve_step(self, step, alpha):
        """
        Parameters: step: int
                        The FOI step for which the solution is desired.
                    alpha: int
                        The FOI associated to step.
        Returns:    y_alpha: array
                        The solution associated to the given step.
        """

        print('Working on alpha = {0}'.format(alpha))
        # Set the progress bar value if one exists.
        if self.sdm.gui != 0:
            # The max value for the current progress updates.
            const = 2 * self.sdm.nraces * self.sdm.sruns
            max_0 = self.mfsg_id + 1.0 / const
            denom = self.t_max * const
            max_1 = max(self.mfsg_id + self.current_t / denom, self.t_step)
            self.t_step = min(max_1, max_0)
            self.set_progress(self.t_step)
        if not self.sdm.scheck:
            # Get the solution for alpha.
            y_alpha =  self.sdm.foi_dist[step] * self.solve_foi(alpha)
        else:
            # Get the array of stochastic solutions.
            y_alpha = []
            for _ in range(self.sdm.sruns):
                print('Stochastic run number: {0}'.format(_))
                y_alpha += [self.sdm.foi_dist[step] * self.solve_sfoi(alpha)]
            y_alpha = np.array(y_alpha)
        return y_alpha

    def theta_function(self, f):
        """
        Parameters: f: list
                        The list of stock counts, ordered according to:
                        [S_1, I_1, U_1, Z_1, ..., S_n, I_n, U_n, Z_n], where
                        for each i in [1, ..., n], and stock X in [S, I, U, Z],
                        X_i is the list of 'X' stocks for alpha = i. For
                        example, with three age groups, S_1 = S_1(0), S_1(1),
                        S_1(2).
        Returns:    t: list
                        The list of theta values for each product.
        """

        t = []
        # Loop through each product and for each alpha, obtain the sum of
        # associated use stocks, weighted by the probability of alpha.
        exp_foi_terms = np.array(self.sdm.foi_range) * self.sdm.foi_dist
        for p in range(self.sdm.sizes['tobacco_products']):
            prod_sums = []
            for alpha in self.sdm.foi_range:
                p_sum = sum(f[x] for x in self.sdm.theta_indices[alpha][p])
                prod_sums += [p_sum]
            # Calculate the weighted sum.
            t += [np.dot(exp_foi_terms, prod_sums) / self.sdm.exp_foi]
        return t

    # Read-only properties.
    @property
    def mfsg_id(self):
        return self._id

    @property
    def sdm(self):
        return self._sdm

    @property
    def sex_race(self):
        return self._sex_race

    @property
    def sol(self):
        return self._sol

    @property
    def sol_high(self):
        return self._sol_high

    @property
    def sol_low(self):
        return self._sol_low

    @property
    def theta(self):
        return self._theta

    # Read-write properties.
    @property
    def current_t(self):
        return self._current_t

    @current_t.setter
    def current_t(self, val):
        if val >= 0:
            self._current_t = val

