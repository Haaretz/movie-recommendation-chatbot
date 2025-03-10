def add(a: float, b: float):
    """returns a + b."""
    print(type(a))
    return a + b


def subtract(a: float, b: float):
    """returns a - b."""
    return a - b


def multiply(a: float, b: float):
    """returns a * b."""
    return a * b


def divide(a: float, b: float):
    """returns a / b."""
    return a / b


def percentage_change(old_value: float, new_value: float):
    """Returns the percentage change from old_value to new_value."""
    if old_value == 0:
        return float("inf")
    change = ((new_value - old_value) / old_value) * 100
    return round(change, 2)


if __name__ == "__main__":
    print(percentage_change(50, 25.0001))
