import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class ModalAnalysis(object):
    def __init__(self, M, K, F):
        self.M = M
        self.K = K
        self.F = F
        self.dof = self.M.shape[0]

    def NaturalValue(self, fname):
        self.fname = fname
        dof = self.dof
        omega, v = LA.eig(np.dot(LA.inv(self.M), self.K))

        # SORT THE EIGENVALUE
        omega_sort = np.sort(omega)
        sort_index = np.argsort(omega)

        # SORT THE EIGENVECTOR
        v_sort = []
        for i in range(len(sort_index)):
            v_sort.append(v.T[sort_index[i]])
        v_sort = np.array(v_sort)
        for i in range(self.dof):
            v_sort[i] /= v_sort[i, 1]
        v_sort_view = v_sort[:, ::-1]

        # PRINT OUT
        self.naturalFreq = np.sqrt(omega_sort)
        self.naturalPeriod = 2 * np.pi / np.sqrt(omega_sort)
        self.eigenVector = v_sort
        print('natural frequency=', self.naturalFreq)
        print('natural period=', self.naturalPeriod)
        print('eigen vector=', self.eigenVector)

        axis = np.linspace(0, len(sort_index), len(sort_index) + 1)
        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(111)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')

        ax1.set_xlabel('Degree of freedom')
        ax1.set_ylabel('Eigenvector')

        ax1.set_xticks(np.arange(0, dof + 1, 1))
        ax1.set_yticks(np.arange(-2, 2, 0.5))
        ax1.set_xlim(0, dof)
        # ax1.set_ylim(-1, 1)

        # PLOT
        vec_label = [
            r"$\omega_{1}$",
            r"$\omega_{2}$",
            r"$\omega_{3}$",
            r"$\omega_{4}$",
            r"$\omega_{5}$"]
        period_label = [
            r"$T_{1}$",
            r"$T_{2}$",
            r"$T_{3}$",
            r"$T_{4}$",
            r"$T_{5}$"]
        for i in range(len(sort_index)):
            eigen_vector = np.concatenate([[0], v_sort_view[i]])
            ax1.plot(
                axis,
                eigen_vector,
                label=vec_label[i] +
                '=' +
                str('{:.1f}'.format(self.naturalFreq[i])) + ',' + period_label[i] +
                '=' +
                str('{:.4f}'.format(self.naturalPeriod[i])),
                lw=1,
                marker='o')

        fig.tight_layout()
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', )
        plt.savefig(
            'fig/%s_modeshape.jpeg' %
            fname,
            bbox_inches="tight",
            pad_inches=0.05)

    def Participation(self):
        eigenVector = self.eigenVector
        print(eigenVector)
        den = np.zeros(self.dof)
        num = np.zeros(self.dof)
        beta = np.zeros(self.dof)

        for i in range(self.dof):
            # eigenVector[i]/=eigenVector[i,0]
            den[i] = np.dot(eigenVector[i], self.F.T)
            num[i] = np.dot(eigenVector[i], np.dot(self.M, eigenVector[i]))
            print(den[i], num[i])
            beta[i] = den[i] / num[i]
        self.beta = beta
        print(den, num)
        print('Participation factor=', beta)
