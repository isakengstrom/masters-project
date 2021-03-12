
# Flags
SHOULD_LIMIT = True # If false, runs through the whole FOI dataset, else it runs through the intervals specified in LIMIT_PARAMS

# Setting a limit to 'None' or '0<' will disable that limit
# These limits work like this: "lower_lim" <= param < "upper_lim"
LIMIT_PARAMS = {
    # Limits for the subjects
    "sub_lower_lim": None,
    "sub_upper_lim": None,
    # Limits for the sessions
    "sess_lower_lim": None,
    "sess_upper_lim": None,
    # Limits for the views
    "view_lower_lim": None,
    "view_upper_lim": None,
    # Limits for the frames
    "frame_lower_lim": None,
    "frame_upper_lim": None,
}


def lower_lim_check(ind, param):
    if LIMIT_PARAMS[param + "_lower_lim"] is None or LIMIT_PARAMS[param + "_lower_lim"] < 0:
        return False
    else:
        return LIMIT_PARAMS[param + "_lower_lim"] > ind


def upper_lim_check(ind, param):
    if LIMIT_PARAMS[param + "_upper_lim"] is None or LIMIT_PARAMS[param + "_upper_lim"] < 0:
        return False
    elif LIMIT_PARAMS[param + "_lower_lim"] > LIMIT_PARAMS[param + "_upper_lim"]:
        return False
    else:
        return LIMIT_PARAMS[param + "_upper_lim"] < ind
