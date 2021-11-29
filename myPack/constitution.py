import enum
import numpy as np
from scipy.fftpack.basic import fft

class State(enum.Enum):
    LINEAR = enum.auto()
    ELASTIC = enum.auto()
    PLASTIC = enum.auto()
    SLIP = enum.auto()

class PN(enum.Enum):
    POSITIVE = 1
    NEGATIVE = -1
    DEFAULT = 0

    @classmethod
    def val(cls,target):
        target = -np.sign(target)
        for pn in cls:
            if target == pn.value:
                return pn


class Linear:
    def __init__(self,k):
        self.x,self.f = [0,0],[0,0]
        self.k = k
        # 更新予定の値たち
        self.next_f,self.next_x = 0,0

    def sheer(self,x):
        self.next_x = x
        self.next_f = self.k*x
        return self.next_f

    def push(self):
        self.x += [self.next_x]
        self.f += [self.next_f]

    def sheer_self(self,x):
        f = self.sheer(x)
        return f,self

    def push_self(self):
        self.push()
        return self

    @property
    def X(self):
        return self.x[2:]

    @property
    def F(self):
        return self.f[2:]

    @property
    def Ktan(self):
        return self.k


class Slip(Linear):
    def __init__(self,k1,k2,dyield):
        self.x = [0,0]
        self.f = [0,0]
        self.k1,self.k2 = k1,k2
        self.dyield = dyield
        self.state = State.LINEAR
        self.pn = PN.DEFAULT
        self.emax,self.emin = dyield,-dyield  # State.ELASTICの限界値

        # 更新予定の値たち
        self.next_f,self.next_x = 0,0
        self.next_state = State.LINEAR
        self.next_pn = PN.DEFAULT
        self.next_emax,self.next_emin = dyield,-dyield

    # スリップのxの範囲
    @property
    def smax(self):
        return self.emax - self.dyield

    @property
    def smin(self):
        return self.emin + self.dyield

    def fstraight(self,k):
        start = self.x[-1]
        end = self.next_x
        return self.f[-1] + k*(end-start)

    def fcorner(self,mid):
        start = self.x[-1]
        end = self.next_x
        if self.state == State.LINEAR or self.state == State.ELASTIC:
            kstart,kend = self.k1,self.k2
        else:
            kstart,kend = self.k2,self.k1
        return self.f[-1] + kstart*(mid-start) + kend*(end-mid)

    def sheer(self,x):
        self.next_x = x
        state = self.state

        if self.state == State.LINEAR:
            if x >= self.emax:
                f = self.fcorner(self.emax)
                state = State.PLASTIC
                self.next_pn = PN.POSITIVE
            elif x <= self.emin:
                f = self.fcorner(self.emin)
                state = State.PLASTIC
                self.next_pn = PN.NEGATIVE
            else:
                f = self.fstraight(self.k1)

        elif self.state == State.ELASTIC:
            if self.pn == PN.POSITIVE:
                if x >= self.emax:
                    f = self.fcorner(self.emax)
                    state = State.PLASTIC
                elif x <= self.smax:
                    f = self.fcorner(self.smax)
                    state = State.SLIP
                    self.next_pn = PN.DEFAULT
                else:
                    f = self.fstraight(self.k1)
            elif self.pn == PN.NEGATIVE:
                if x <= self.emin:
                    f = self.fcorner(self.emin)
                    state = State.PLASTIC
                elif x >= self.smin:
                    f = self.fcorner(self.smin)
                    state = State.SLIP
                    self.next_pn = PN.DEFAULT
                else:
                    f = self.fstraight(self.k1)

        elif self.state == State.PLASTIC:
            dx = x - self.x[-1]
            if dx*self.pn.value < 0:  # reverse
                if dx > 0:
                    self.next_emin = self.x[-1]
                else:
                    self.next_emax = self.x[-1]
                f = self.fstraight(self.k1)
                state = State.ELASTIC
            else:
                f = self.fstraight(self.k2)

        elif self.state == State.SLIP:
            if x >= self.smax:
                f = self.fcorner(self.smax)
                state = State.ELASTIC
                self.next_pn = PN.POSITIVE
            elif x <= self.smin:
                f = self.fcorner(self.smin)
                state = State.ELASTIC
                self.next_pn = PN.NEGATIVE
            else:
                f = self.fstraight(self.k2)

        self.next_f = f
        self.next_state = state
        return f

    def push(self):
        self.x += [self.next_x]
        self.f += [self.next_f]
        self.state = self.next_state
        self.pn = self.next_pn
        self.emax = self.next_emax
        self.emin = self.next_emin

    @property
    def Ktan(self):
        state = self.next_state
        if state == State.LINEAR or state == State.ELASTIC:
            return self.k1
        else:
            return self.k2


class Bilinear(Slip):
    def __init__(self, k1, k2, dyield):
        super().__init__(k1, k2, dyield)
        self.state,self.next_state = State.ELASTIC,State.ELASTIC

    @property
    def emax(self):
        return self._emax

    @emax.setter
    def emax(self,value):
        if value is not None:
            self._emax = value
            self._emin = value - 2*self.dyield

    @property
    def emin(self):
        return self._emin

    @emin.setter
    def emin(self,value):
        if value is not None:
            self._emin = value
            self._emax = value + 2*self.dyield

    @property
    def next_emax(self):
        return self._next_emax

    @next_emax.setter
    def next_emax(self,value):
        self._next_emax = value
        self._next_emin = None

    @property
    def next_emin(self):
        return self._next_emin

    @next_emin.setter
    def next_emin(self,value):
        self._next_emin = value
        self._next_emax = None

    def sheer(self, x):
        self.next_x = x
        state = self.state

        if self.state == State.ELASTIC:
            if x >= self.emax:
                f = self.fcorner(self.emax)
                state = State.PLASTIC
                self.next_pn = PN.POSITIVE
            elif x <= self.emin:
                f = self.fcorner(self.emin)
                state = State.PLASTIC
                self.next_pn = PN.NEGATIVE
            else:
                f = self.fstraight(self.k1)

        elif self.state == State.PLASTIC:
            dx = x - self.x[-1]
            if dx*self.pn.value < 0:  # reverse
                if dx > 0:
                    self.next_emin = self.x[-1]
                else:
                    self.next_emax = self.x[-1]
                f = self.fstraight(self.k1)
                state = State.ELASTIC
            else:
                f = self.fstraight(self.k2)

        self.next_f = f
        self.next_state = state
        return f

    @property
    def Ktan(self):
        state = self.next_state
        if state == State.ELASTIC:
            return self.k1
        else:
            return self.k2


class Combined:
    def __init__(self,model):
        # model: list of constitution models
        # example: [[Linear(*args)], [Slip(*args), Bilinear(*args)]]
        # This means 1st-floor uses model Linear(*args) and 2nd-floor uses Slip(*args) AND Bilinear(*args).
        # If more than one model is in one floor, the sheer force will be added.
        self.model = model

    def sheer(self,x):
        f = []
        for floor,x_floor in zip(self.model,x):
            f_floor = 0
            for part in floor:
                f_floor += part.sheer(x_floor)
            f += [f_floor]
        return np.array(f)

    def push(self):
        for floor in self.model:
            for part in floor:
                part.push()

    @property
    def Ktan(self):
        K = []
        for floor in self.model:
            K_floor = 0
            for part in floor:
                K_floor += part.Ktan
            K += [K_floor]
        return np.array(K)

    @property
    def X(self):
        Xlist = []
        for floor in self.model:
            for part in floor:
                Xlist += [part.X]
        return np.stack(Xlist)

    @property
    def F(self):
        Flist = []
        for floor in self.model:
            for part in floor:
                Flist += [part.F]
        return np.stack(Flist)
