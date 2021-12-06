from enum import Enum,auto
import numpy as np

class PN(Enum):
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
    class State(Enum):
        LINEAR = auto()
        ELASTIC = auto()
        PLASTIC = auto()
        ULTIMATE = auto()
        SLIP = auto()


    def __init__(self,k1,k2,dyield):
        State = self.State
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
        State = self.State
        start = self.x[-1]
        end = self.next_x
        if self.state == State.LINEAR or self.state == State.ELASTIC:
            kstart,kend = self.k1,self.k2
        else:
            kstart,kend = self.k2,self.k1
        return self.f[-1] + kstart*(mid-start) + kend*(end-mid)

    def sheer(self,x):
        State = self.State
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
        State = self.State
        state = self.next_state
        if state == State.LINEAR or state == State.ELASTIC:
            return self.k1
        else:
            return self.k2


class Bilinear(Slip):
    class State(Enum):
        LINEAR = auto()
        ELASTIC = auto()
        PLASTIC = auto()

    def __init__(self,k1,k2,dyield):
        State = self.State
        super().__init__(k1,k2,dyield)
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
        State = self.State
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
        State = self.State
        state = self.next_state
        if state == State.ELASTIC:
            return self.k1
        else:
            return self.k2


class Slip_Bilinear(Linear):
    def __init__(self,k,alpha,dyield,slip_rate):
        self.x,self.f = [0,0],[0,0]
        slip_args = {'k1':k*slip_rate,'k2':k*alpha,'dyield':dyield}
        bilin_args = {'k1':k*(1-slip_rate),'k2':0,'dyield':dyield}
        self.slip = Slip(**slip_args)
        self.bilinear = Bilinear(**bilin_args)
        self.next_f,self.next_x = 0,0

    def sheer(self,x):
        self.next_x = x
        self.next_f = self.slip.sheer(x)+self.bilinear.sheer(x)
        return self.next_f

    def push(self):
        self.slip.push()
        self.bilinear.push()
        return super().push()

    @property
    def Ktan(self):
        return self.slip.Ktan+self.bilinear.Ktan


class Slip2(Slip):
    class State(Enum):
        LINEAR = auto()
        ELASTIC = auto()
        PLASTIC = auto()
        ULTIMATE = auto()
        SLIP = auto()

    def __init__(self,k1,k2,k3,d1,d2):
        State = self.State
        self.x,self.f = [0,0],[0,0]
        self.k1,self.k2,self.k3 = k1,k2,k3
        self.d1,self.d2 = d1,d2
        self.state = State.LINEAR
        self.pn = PN.DEFAULT
        self.emax,self.emin = 0,0
        self.fmax,self.fmin = 0,0
        self.skeleton_p,self.skeleton_n = (self.state,k1),(self.state,k1)

        self.next_f,self.next_x = 0,0
        self.next_state = State.LINEAR
        self.next_pn = PN.DEFAULT
        self.next_emax,self.next_emin = 0,0
        self.next_fmax,self.next_fmin = 0,0
        self.next_skeleton_p,self.next_skeleton_n = (self.state,k1),(self.state,k1)

    @property
    def smax(self):
        return self.emax - self.fmax/self.k1

    @property
    def smin(self):
        return self.emin - self.fmin/self.k1

    def fstraight(self, k):
        return super().fstraight(k)

    def fcorner(self,mid,kstart,kend):
        start = self.x[-1]
        end = self.next_x
        return self.f[-1] + kstart*(mid-start) + kend*(end-mid)

    def sheer(self,x):
        State = self.State
        self.next_x = x
        state = self.state
        k1,k2,k3 = self.k1,self.k2,self.k3

        if state == State.LINEAR:
            if x >= self.d1:
                f = self.fcorner(self.d1,k1,k2)
                state = State.PLASTIC
                self.next_pn = PN.POSITIVE
                self.next_skeleton_p = state,k2
            elif x <= -self.d1:
                f = self.fcorner(-self.d1,k1,k2)
                state = State.PLASTIC
                self.next_pn = PN.NEGATIVE
                self.next_skeleton_n = state,k2
            else:
                f = self.fstraight(k1)

        elif state == State.PLASTIC:
            dx = x - self.x[-1]
            if dx*self.pn.value < 0:  # reverse
                if self.pn == PN.POSITIVE:
                    self.next_emax = self.x[-1]
                    self.next_fmax = self.f[-1]
                else:
                    self.next_emin = self.x[-1]
                    self.next_fmin = self.f[-1]
                f = self.fstraight(k1)
                state = State.ELASTIC
            elif x >= self.d2:
                f = self.fcorner(self.d2,k2,k3)
                state = State.ULTIMATE
                self.next_skeleton_p = state,k3
            elif x <= -self.d2:
                f = self.fcorner(-self.d2,k2,k3)
                state = State.ULTIMATE
                self.next_skeleton_n = state,k3
            else:
                f = self.fstraight(k2)

        elif state == State.ULTIMATE:
            dx = x - self.x[-1]
            if dx*self.pn.value < 0:  # reverse
                if self.pn == PN.POSITIVE:
                    self.next_emax = self.x[-1]
                    self.next_fmax = self.f[-1]
                else:
                    self.next_emin = self.x[-1]
                    self.next_fmin = self.f[-1]
                f = self.fstraight(self.k1)
                state = State.ELASTIC
            else:
                f = self.fstraight(k3)
                if f*self.pn.value < 0:
                    f = 0


        elif state == State.ELASTIC:
            if self.pn == PN.POSITIVE:
                if x >= self.emax:
                    state,k = self.skeleton_p
                    f = self.fcorner(self.emax,k1,k)
                elif x <= self.smax:
                    f = self.fcorner(self.smax,k1,0)
                    state = State.SLIP
                    self.next_pn = PN.DEFAULT
                else:
                    f = self.fstraight(k1)
            elif self.pn == PN.NEGATIVE:
                if x <= self.emin:
                    state,k = self.skeleton_n
                    f = self.fcorner(self.emin,k1,k)
                elif x >= self.smin:
                    f = self.fcorner(self.smin,k1,0)
                    state = State.SLIP
                    self.next_pn = PN.DEFAULT
                else:
                    f = self.fstraight(k1)

        elif state == State.SLIP:
            if x >= self.smax:
                f = self.fcorner(self.smax,0,k1)
                state = State.ELASTIC
                self.next_pn = PN.POSITIVE
            elif x <= self.smin:
                f = self.fcorner(self.smin,0,k1)
                state = State.ELASTIC
                self.next_pn = PN.NEGATIVE
            else:
                f = self.fstraight(0)

        self.next_f = f
        self.next_state = state
        return f

    def push(self):
        super().push()
        self.fmax,self.fmin = self.next_fmax,self.next_fmin
        self.skeleton_p,self.skeleton_n = self.next_skeleton_p,self.next_skeleton_n

    @property
    def Ktan(self):
        State = self.State
        state = self.next_state
        if state == State.LINEAR or state == State.ELASTIC:
            return self.k1
        elif state == State.PLASTIC:
            return self.k2
        elif state == State.ULTIMATE:
            return self.k3
        else:
            return 0


class Slip_Bilinear2(Slip_Bilinear):
    def __init__(self,k,h):
        fyield = k*h/285.8
        d1 = 7e-4*h
        d2 = h/120
        d3 = h/30
        kb1 = 190.5*fyield/h
        kb2 = 9.53*fyield/h
        ks1 = 95.3*fyield/h
        ks2 = 22.5*fyield/h
        ks3 = -31.1*fyield/h

        self.x,self.f = [0,0],[0,0]
        self.slip = Slip2(ks1,ks2,ks3,d2,d3)
        self.bilinear = Bilinear(kb1,kb2,d1)
        self.next_f,self.next_f = 0,0


class Slip_Bilinear3(Slip_Bilinear):
    def __init__(self,k,h):
        d1 = 0.5e-2*h
        d2 = 1.5e-2*h
        d3 = h
        k1,k3 = k,0.14*k
        kb1 = k3
        kb2 = 0.4*k3
        ks1 = k1-kb1
        ks2 = k3-kb2
        ks3 = 0

        self.x,self.f = [0,0],[0,0]
        self.slip = Slip2(ks1,ks2,ks3,d1,d3)
        self.bilinear = Bilinear(kb1,kb2,d2)
        self.next_f,self.next_f = 0,0


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
