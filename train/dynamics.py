import math


class Dynamics:
    def __init__(self) -> None:
        self.epoch=0 # current epoch
        self.__b_loss = [] # batch loss
        self.__iteration = 0 # current iteration



    @property
    def loss(self):
        return sum(self.__b_loss) / len(self.__b_loss)

    @loss.setter
    def loss(self, value):
        raise Exception("read only")



    @property
    def b_loss(self):
        return self.__b_loss[-1]

    @b_loss.setter
    def b_loss(self, value):
        self.__b_loss.append(value)



    @property
    def iteration(self):
        return self.__iteration

    @iteration.setter
    def iteration(self, value):
        if value == 0:
            self.__b_loss = []
        self.__iteration = value
