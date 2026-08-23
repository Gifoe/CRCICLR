mods = ['pyarrow', 'fastparquet', 'polars', 'duckdb', 'mne']
for name in mods:
    try:
        module = __import__(name)
        print(name, getattr(module, '__version__', 'available'))
    except Exception as exc:
        print(name, type(exc).__name__)
