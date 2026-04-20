---
marp: true
---

# Module 7: Lists & History 📋

## Storing Multiple Values

**Duration: 7 minutes | Time: 1:03-1:10**

---

## Why Single Variables Aren't Enough 🤔

### The Problem

```python
calculation1 = "5 + 3 = 8"
calculation2 = "10 - 2 = 8"
calculation3 = "4 * 6 = 24"
# What if user does 100 calculations?
# We can't create 100 variables!
```

### The Solution: Lists

```python
history = ["5 + 3 = 8", "10 - 2 = 8", "4 * 6 = 24"]
# One variable holds many values!
```

---

## What Are Lists? 📦

### Think of Lists as...

- 📚 **Bookshelf** holding multiple books
- 🎵 **Playlist** with many songs
- 📝 **Shopping list** with many items
- 💾 **Memory bank** for your calculator

### List Characteristics

- **Ordered** - items have positions (0, 1, 2...)
- **Mutable** - can add, remove, change items
- **Mixed types** - can hold numbers, strings, etc.
- **Dynamic** - grows and shrinks as needed

---

## Creating Lists 🛠️

### Empty Lists

```python
history = []                    # Empty list
numbers = list()               # Another way to create empty list
```

### Lists with Initial Values

```python
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, 3.14, True]
```

### Calculator History Example

```python
calculations = [
    "5 + 3 = 8",
    "10 - 2 = 8",
    "15 / 3 = 5.0"
]
```

---

## Adding Items to Lists ➕

### `append()` - Add to End

```python
history = []
history.append("5 + 3 = 8")
history.append("10 - 2 = 8")
print(history)  # ["5 + 3 = 8", "10 - 2 = 8"]
```

### `insert()` - Add at Specific Position

```python
fruits = ["apple", "orange"]
fruits.insert(1, "banana")  # Insert at position 1
print(fruits)  # ["apple", "banana", "orange"]
```

### Calculator Example

```python
def add_calculation(calc_string):
    history.append(calc_string)
    print(f"Added: {calc_string}")
```

---

## Accessing List Items 🎯

### Index Numbers (Start at 0!)

```python
fruits = ["apple", "banana", "orange"]

first = fruits[0]      # "apple"
second = fruits[1]     # "banana"
last = fruits[2]       # "orange"
```

### Negative Indices (Count from End)

```python
fruits = ["apple", "banana", "orange"]

last = fruits[-1]      # "orange"
second_last = fruits[-2]  # "banana"
```

### Getting List Length

```python
count = len(fruits)    # 3
```

---

## Looping Through Lists 🔄

### Basic Loop Pattern

```python
history = ["5 + 3 = 8", "10 - 2 = 8", "15 / 3 = 5.0"]

for calculation in history:
    print(calculation)
```

### Loop with Index Numbers

```python
for i in range(len(history)):
    print(f"{i + 1}. {history[i]}")

# Output:
# 1. 5 + 3 = 8
# 2. 10 - 2 = 8
# 3. 15 / 3 = 5.0
```

### Enumerate (Easier Way)

```python
for i, calculation in enumerate(history, 1):
    print(f"{i}. {calculation}")
```

---

## 🔨 Live Coding: Calculator v2.1

### Adding Memory to Our Calculator!

```python
history = []  # Global list to store calculations

def add_to_history(calc_string):
    history.append(calc_string)

def show_history():
    print("\n📋 CALCULATION HISTORY")
    if history:
        for i, calc in enumerate(history, 1):
            print(f"{i}. {calc}")
    else:
        print("No calculations yet!")

# In your calculator loop:
# After doing a calculation:
calc_string = f"{num1} {op} {num2} = {result}"
add_to_history(calc_string)

# Add to menu: "3️⃣ Show History"
```

---

## What We Just Added 🎉

### New Features

- ✅ **Memory system** stores all calculations
- ✅ **History display** shows numbered list
- ✅ **Persistent storage** across multiple calculations
- ✅ **Professional formatting** with numbers

### Functions We Created

- `add_to_history()` - Saves calculations
- `show_history()` - Displays all history
- Demonstrates **code organization**

---

## Quick Exercise (2 minutes) ⚡

### Your Turn!

1. **Add history option** to your calculator menu
2. **Perform several calculations** and check history
3. **Think about clearing history** - how would you do it?

### Challenge Ideas

```python
def clear_history():
    history.clear()
    print("🗑️ History cleared!")

def get_last_calculation():
    if history:
        return history[-1]
    return "No calculations yet"
```

---

## Useful List Methods 🛠️

### Adding Items

```python
my_list = [1, 2, 3]
my_list.append(4)        # [1, 2, 3, 4]
my_list.extend([5, 6])   # [1, 2, 3, 4, 5, 6]
```

### Removing Items

```python
my_list = ["a", "b", "c", "b"]
my_list.remove("b")      # Removes first "b"
item = my_list.pop()     # Removes and returns last item
item = my_list.pop(0)    # Removes and returns first item
```

### Checking Contents

```python
fruits = ["apple", "banana", "orange"]
has_apple = "apple" in fruits        # True
apple_count = fruits.count("apple")  # 1
```

---

## List Patterns for Calculators 💡

### Recent Calculations

```python
def get_recent_history(count=5):
    return history[-count:]  # Last 5 calculations
```

### Find Calculations by Operation

```python
def find_additions():
    additions = []
    for calc in history:
        if " + " in calc:
            additions.append(calc)
    return additions
```

### Statistics

```python
def get_stats():
    total_calculations = len(history)
    if total_calculations > 0:
        print(f"Total calculations: {total_calculations}")
        print(f"Most recent: {history[-1]}")
```

---

## Common List Mistakes 🚨

### Index Out of Range

```python
my_list = [1, 2, 3]
item = my_list[5]  # IndexError: list index out of range
```

### Modifying List While Looping

```python
# Dangerous!
for item in my_list:
    if item == "bad":
        my_list.remove(item)  # Can skip items!
```

### Forgetting Lists Are Objects

```python
list1 = [1, 2, 3]
list2 = list1        # Both point to same list!
list2.append(4)
print(list1)         # [1, 2, 3, 4] - list1 changed too!
```

---

## List comprehension

It is a concise way to create lists in Python. It lets you build a new list by applying an expression to each item in an existing sequence, all in a single line of code.

The basic syntax is:

```python
[expression for item in iterable]
```

Instead of writing:

```python
squares = []
for x in range(5):
    squares.append(x**2)
```

You can simply write: `squares = [x**2 for x in range(5)]`

---

## List comprehension

Here are some practical examples:

**Creating a list of squares:**

```python
squares = [x**2 for x in range(5)]
# Result: [0, 1, 4, 9, 16]
```

**With a condition (filtering):**

```python
evens = [x for x in range(10) if x % 2 == 0]
# Result: [0, 2, 4, 6, 8]
```

**Transforming strings:**

```python
names = ['alice', 'bob', 'charlie']
capitalized = [name.upper() for name in names]
# Result: ['ALICE', 'BOB', 'CHARLIE']
```

**More complex example:**

```python
# Get lengths of words that start with 'a'
words = ['apple', 'banana', 'apricot', 'cherry']
lengths = [len(word) for word in words if word.startswith('a')]
# Result: [5, 7]
```

---

## Lists vs Other Data Types 📊

### Lists vs Strings

```python
# String
word = "hello"
word[0] = "H"  # Error! Strings are immutable

# List
letters = ["h", "e", "l", "l", "o"]
letters[0] = "H"  # Works! Lists are mutable
```

### When to Use Lists

- ✅ **Multiple related items** (calculations, names, scores)
- ✅ **Order matters** (first calculation, second calculation)
- ✅ **Need to add/remove** items dynamically
- ✅ **Unknown final size** (user might do 1 or 100 calculations)

---

## Key Takeaways 📚

### What You Learned

- ✅ **Lists store multiple values** in one variable
- ✅ **append()** adds items to end
- ✅ **Index access** with [0], [1], etc.
- ✅ **enumerate()** for numbered loops
- ✅ **History tracking** in calculators

### Next Up: Dictionaries!

- Store key-value pairs
- User settings and preferences
- Operation name lookups
- More advanced data organization

**Your calculator has memory! 🧠**
