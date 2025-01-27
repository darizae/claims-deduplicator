from typing import List


def select_longest(claims: List[str]) -> str:
    return max(claims, key=len)


def select_shortest(claims: List[str]) -> str:
    return min(claims, key=len)


def select_random(claims: List[str]) -> str:
    import random
    return random.choice(claims)
