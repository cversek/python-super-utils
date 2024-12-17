# python-super-utils

**`python-super-utils`** is a Python package providing advanced debugging tools and customizable tagged logging for developers.

- **Debugging utilities**: Print detailed traceback, embed IPython shells, and handle Matplotlib plots in non-blocking mode.
- **Tagged logging**: Automatically include module and function names in log outputs for better traceability.

---

## **Features**

### Debugging Tools
- `DEBUG_TAG`: Print the file name, line number, and optional messages for debugging checkpoints.
- `DEBUG_PRINT_EXCEPTION`: Display formatted exception tracebacks.
- `DEBUG_EMBED`: Drop into an IPython shell with the current namespace for live debugging.
- `DEBUG_PLOTS_NONBLOCKING`: Show Matplotlib plots in a non-blocking way.

### Tagged Logging
- Logs include module and function names as tags for easier traceability.
- Integrates seamlessly with the Python `logging` module.

---

## **Installation**

### Standard Installation (End-Users)
Install the latest version directly from PyPI:
```bash
pip install python-super-utils
```

If the package is not yet published on PyPI, you can install it directly from the GitHub repository:
```bash
pip install git+https://github.com/cversek/python-super-utils.git
```

### Developer Installation (Editable Mode)
For developers contributing to or testing the project, install in editable mode:

```bash
git clone https://github.com/cversek/python-super-utils.git
cd python-super-utils
pip install -e .
```

Editable installation allows changes to the source code to reflect immediately without reinstalling the package.

---

## **Usage**

### Debugging Tools
Example: DEBUG_TAG
Print the file name, line number, and optional message.
```python
from super_utils.debug import DEBUG_TAG, currentframe

def example():
    DEBUG_TAG(currentframe(), "This is a debug checkpoint")

example()
```
Output:

```
****************************************
*** DEBUG ***
***     L#6 in 'example.py': This is a debug checkpoint
****************************************
```
Note: The line number and file name are automatically extracted from the current frame. Unfortunately, this cannot be done inside the library, so you need to pass the current frame to the function.

Example: DEBUG_PRINT_EXCEPTION
Print a formatted exception traceback.
```python
from super_utils.debug import DEBUG_PRINT_EXCEPTION

try:
    1 / 0
except:
    DEBUG_PRINT_EXCEPTION()
```
Output:
```
****************************************
*** EXCEPTION ***
*** 
    Traceback (most recent call last):
      File "example.py", line 4, in <module>
        1 / 0
    ZeroDivisionError: division by zero
****************************************
```

Example: DEBUG_EMBED
Drop into an IPython shell with the current namespace for live debugging.
```python
from super_utils.debug import DEBUG_EMBED

global_variable = "Hello, Global Debug!"

def interactive_debug():
    local_variable = "Hello, Local Debug!"
    DEBUG_EMBED(local_ns=locals(), global_ns=globals())

interactive_debug()
```
Note: We have to capture the namespaces manually.  The `global_ns` argument is optional and can be omitted if you only need to debug local variables.

**Power User Tip:**
This multi-statement line is the most common way to use the debugging tools:
```python
DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals(),exit=True)
```
  The exit=True argument will exit the program after the shell is closed.  I like to copy/paste this liberally into my code, and comment out the lines I don't need.  Here's an example:    


```python
from super_utils.debug import DEBUG_TAG, DEBUG_EMBED

def interactive_debug():
    a = 1
    b = 2
    DEBUG_TAG(currentframe());DEBUG_EMBED(local_ns=locals(),global_ns=globals(),exit=True)
    print("This should not be printed")

interactive_debug()
```

### Tagged Logging
Example: Tagged Logs
```python
from super_utils.logging import setup_tagged_logger

logger = setup_tagged_logger()

def example_function():
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")

example_function()

```
Output:
```
[example_function] DEBUG: This is a debug message.
[example_function] INFO: This is an info message.
```
---
## **Examples**
The examples/ directory contains scripts demonstrating all functionalities:
- `example_debug.py`: Debugging tools examples.
- `example_logging.py`: Tagged logging examples.
- `example_combined.py`: Combined usage of debugging and logging utilities.
To run the examples, navigate to the examples directory and run:
```bash
cd examples
python example_debug.py
python example_logging.py
python example_combined.py
```
---

## **Contributing**
Contributions are welcome! Follow these steps:
1. Fork the repository on GitHub
2. Clone the forked repository to your local machine
```bash
git clone https://github.com/your-username/python-super-utils.git
```
3. Install the package in editable mode
```bash
pip install -e .
```
4. Create a new branch, make your changes, and submit a pull request.


## **License**
This project is open-sourced under the MIT License - see the LICENSE file for details.

## **Links**
- [GitHub Repository](https://github.com/cversek/python-super-utils)
- [Issues](https://github.com/cversek/python-super-utils/issues)