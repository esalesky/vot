import numpy as np

def logsumexp(ns):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)


class MRF(object):

    def __init__(self, num_vowels=9):
        self.calV = range(num_vowels)
        # non-negative unary potential
        self.phi = np.random.randn(len(self.calV))
        # non-negative binary potential
        self.psi = np.random.randn(len(self.calV), len(self.calV))

        print(self.logZ())
        dphi, dpsi = self.dlogZ(np.zeros_like(self.phi), np.zeros_like(self.psi))
        print(dphi)
        print(self.fd())
        exit(0)
        
        # check that it sums 1
        logZ = self.logZ()
        total = -np.inf
        for V in self.subsets(self.calV):
            total = logsumexp(np.asarray([total, self.logp(V)]))
        print(total)

    def subsets(self, s):
        """
        https://codereview.stackexchange.com/questions/147633/get-subsets-of-a-set-given-as-a-list
        """
        sets = []
        for i in range(2**len(s)):
            subset = []
            for j in range(len(s)):
                if i & (1 << j) > 0:
                    subset.append(s[j])
            sets.append(subset)
        return sets
            
    def score(self, V):
        """ V is a list of vowels, represented as integers """
        score = 0.0
        for i in V:
            for j in V:
                score += self.psi[i, j]
        for i in V:
            score += self.phi[i]
        return score

    def logp(self, V):
        return self.score(V) - self.logZ()

    def dlogp(self, V):
        """
        TODO: Liz
        """
        pass
    
    def logZ(self):
        """ enumerate all sets """
        logZ = -np.inf
        for V in self.subsets(self.calV):
            logZ = logsumexp(np.asarray([logZ, self.score(V)]))
        return logZ

    def dlogZ(self, dphi, dpsi):
        """ compute gradient of the log partition function """
        logZ = self.logZ()
        for V in self.subsets(self.calV):
            p_V = np.exp(self.score(V) - logZ)
            dphi += p_V * self.phi
        return dphi, dpsi

    def fd(self, eps=1e-5):
        """ 
        write a function to check the gradient 
        """
        dphi = np.zeros((len(self.calV)))
        for i in xrange(len(self.calV)):
            self.phi[i] += eps
            val1 = self.logZ()
            self.phi[i] -= 2*eps
            val2 = self.logZ()
            self.phi[i] += eps
            dphi[i] = (val2-val1)/(2*eps)
        return dphi

    def train(self):
        """
        TODO: Liz

        Estimate parameters
        """
        pass

mrf = MRF()    
