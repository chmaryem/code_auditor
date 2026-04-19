def calculate_sum(a, b):
    """Calculates the sum of two numbers."""
    return a + b

def greet(name):
    """Greets the user."""
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello, World!")

if __name__ == "__main__":
    result = calculate_sum(5, 10)
    print(f"Result: {result}")
    greet("Alice")
