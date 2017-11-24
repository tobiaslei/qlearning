"""

Ref:
http://quantsoftware.gatech.edu/Qlearning_robot
http://mnemstudio.org/path-finding-q-learning-tutorial.htm

Dyna: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

https://classroom.udacity.com/courses/ud501/lessons/5326212698/concepts/54629888650923
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        '''
        @num_states: integer, the number of states to consider
        @num_actions: integer, the number of actions available.
        @alpha: float, the learning rate used in the update rule.
                Should range between 0.0 and 1.0 with 0.2 as a typical value.
        @gamma: float, the discount rate used in the update rule.
                Should range between 0.0 and 1.0 with 0.9 as a typical value.
        @rar: float, random action rate: the probability of selecting a random
                action at each step. Should range between 0.0 (no random actions)
                to 1.0 (always random action) with 0.5 as a typical value.
        @radr: float, random action decay rate, after each update,
                rar = rar * radr. Ranges between 0.0 (immediate decay to 0)
                and 1.0 (no decay). Typically 0.99.
        @dyna: integer, conduct this number of dyna updates for each regular
                update. When Dyna is used, 200 is a typical value.
        @verbose: boolean, if True, your class is allowed to print debugging
                statements, if False, all printing is prohibited.
        '''

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0

        self.learning_rate = alpha
        self.discount_rate = gamma
        self.random_action_rate = rar
        self.random_action_decay_rate = radr
        self.qtable = np.zeros((self.num_states, self.num_actions))
        # init with random small numbers between 0 and 0.1
        # self.qtable = np.random.uniform(0.0, 0.1, (self.num_states, self.num_actions))

        # for Dyna-Q Learning
        self.dyna_num = dyna
        self.transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.visited_s_a = np.zeros((self.num_states, self.num_actions))
        self.reward_model = np.full((self.num_states, self.num_actions), 0.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        if rand.random() <= self.random_action_rate:
            action = rand.randint(0, self.num_actions-1)
        else:
            # max action for state
            # SLOW: action = np.argmax([self.qtable[s_prime][a_prime] for a_prime in range(self.num_actions)])
            action = np.argmax(self.qtable[s])
        self.s = s
        if self.verbose: print "s =", s, "a =", action
        return action

    def _generate_dyna_experience_tuple(self):
        s = rand.randint(0, self.num_states-1)
        a = rand.randint(0, self.num_actions-1)
        # TODO: convert to probability form?
        # s_prime = np.argmax(self.transition_counts[s, a, ])
        sum_p = np.sum(self.transition_counts[s, a, ])
        all_states = range(self.num_states)
        weights = self.transition_counts[s, a, ] / sum_p
        s_prime = np.random.choice(all_states, p=weights)
        r = self.reward_model[s, a]
        return s, a, s_prime, r

    def _update_qtable(self, s, a, s_prime, r):
        self.qtable[s, a] = (1.0 - self.learning_rate) * self.qtable[s, a] + \
                            self.learning_rate * (r + self.discount_rate * np.max(self.qtable[s_prime, ]))

    def _sample_a_state(self, s, a):
        probs = self.transition_counts[s, a] / np.sum(self.transition_counts[s, a])
        sample_result = np.random.multinomial(1, probs, size=1)
        # return np.nonzero(sample_result)[1][0]
        return np.argmax(sample_result)

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action

        Given Experience tuple <s, a, s_prime, r>
        """

        if rand.random() <= self.random_action_rate:
            action = rand.randint(0, self.num_actions - 1)
            self.random_action_rate = self.random_action_rate * self.random_action_decay_rate
        else:
            # max action for s_prime
            # SLOW: action = np.argmax([self.qtable[s_prime][a_prime] for a_prime in range(self.num_actions)])
            action = np.argmax(self.qtable[s_prime, ])

        self._update_qtable(self.s, self.a, s_prime, r)



        """ Dyna-Q """
        # Update transition model
        if self.visited_s_a[self.s, self.a] == 0:
            self.visited_s_a[self.s, self.a] = 1
            # assign small counts to unseen states
            self.transition_counts[self.s, self.a] = np.full(self.num_states, 0.00001)
            self.transition_counts[self.s, self.a, s_prime] += 1
        else:
            self.transition_counts[self.s, self.a, s_prime] += 1

        # Update reward model
        self.reward_model[self.s, self.a] = (1.0 - self.learning_rate) * self.reward_model[self.s, self.a] + self.learning_rate * r


        ### Hallucinate some experience tuples
        known_exp = np.nonzero(self.visited_s_a)
        # vec_hallu = np.vectorize(self._hallucinate)
        if self.dyna_num > 0 and len(known_exp[0]) > 1:
            # randomly pick an expierence tuple
            picked_idxs = np.random.randint(0, len(known_exp[0]) - 1, self.dyna_num)
            dyna_s = known_exp[0][picked_idxs]
            dyna_a = known_exp[1][picked_idxs]
            # dyna_s = np.random.randint(0, self.num_states - 1)
            # dyna_a = np.random.randint(0, self.num_actions - 1)
            vec_sample = np.vectorize(self._sample_a_state)
            dyna_s_prime = vec_sample(dyna_s, dyna_a)
            dyna_r = self.reward_model[dyna_s, dyna_a]

            vec_update_qtable = np.vectorize(self._update_qtable)
            vec_update_qtable(dyna_s, dyna_a, dyna_s_prime, dyna_r)
            # for i in range(self.dyna_num):
            #     # print dyna_s[i], dyna_a[i], dyna_s_prime[i], dyna_r[i]
            #     self._update_qtable(dyna_s[i], dyna_a[i], dyna_s_prime[i], dyna_r[i])


        # Move on the next state
        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime, "a =", action, "r =", r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
