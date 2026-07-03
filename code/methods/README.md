# Methods

This directory contains CleanCore and the baseline implementations used by the lab runners.

All methods expose:

```python
run(ctx: MethodContext) -> MethodOutput
```

`run.py` uses this interface to execute a method, train or evaluate the classifier as needed, and write the final metrics in a common format.
