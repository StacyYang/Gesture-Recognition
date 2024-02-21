from typing import List
from enum import IntEnum

class Classes(IntEnum):
    abort = 0
    circle = 1
    hello = 2
    no = 3
    stop = 4
    turn = 5
    turn_left = 6
    turn_right = 7
    warn = 8

    @staticmethod
    def len() -> int:
        return 9

    @staticmethod
    def names() -> List[str]:
        return [x.name for x in Classes]

    @staticmethod
    def values() -> List[int]:
        return [x.value for x in Classes]


class SequenceClasses(IntEnum):
    abort_hello = 0
    circle_turn = 1
    hello_abort = 2
    no_stop = 3
    stop_no = 4
    turn_circle = 5
    turn_left_turn_right = 6
    turn_right_turn_left = 7

    @staticmethod
    def len() -> int:
        return 8

    @staticmethod
    def names() -> List[str]:
        return [x.name for x in SequenceClasses]

    @staticmethod
    def values() -> List[int]:
        return [x.value for x in SequenceClasses]
