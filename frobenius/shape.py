class Shape(tuple):
    def __new__(cls, *args):
        if len(args) != 2:
            raise TypeError("Shape should have to int arguments")
        return tuple.__new__(cls, (args[0], args[1]))

    @property
    def nrows(self) -> int:
        return self[0]

    @property
    def ncols(self) -> int:
        return self[1]
