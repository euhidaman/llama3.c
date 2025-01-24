import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        config_file = arg
        with open(config_file) as f:
            exec(f.read())
    else:
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
