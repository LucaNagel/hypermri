"""
Generic object manipulation utilities.

Contains functions for managing attributes and counters on arbitrary objects.
"""


def _init_attr(scan=None, attr_name: str = None, attr_type=dict):
    """
    Initialize an attribute on the data object if it doesn't exist.

    Parameters
    ----------
    scan : object
        The object to modify.
    attr_name : str
        The name of the attribute to initialize.
    attr_type : type
        The type of the attribute (default: dict).

    Returns
    -------
    bool
        True if initialization ran (or attribute existed).
    """
    if scan is None or attr_name is None:
        return False

    if not hasattr(scan, attr_name):
        setattr(scan, attr_name, attr_type())
    return True


def add_counter(scan=None, counter_name=None, n_counts=1):
    """
    Add count to a counter attribute on the object, keyed by object ID.

    Parameters
    ----------
    scan : object
        The object to modify.
    counter_name : str
        The name of the counter attribute.
    n_counts : int
        Number to add to the counter.

    Returns
    -------
    int
        The new count value.
    """
    if scan is None:
        raise Exception("scan can not be None!")
    if counter_name is None:
        raise Exception("counter_name can not be None!")

    obj_id = str(id(scan))
    _init_attr(scan=scan, attr_name=counter_name)

    attr = getattr(scan, counter_name)

    if isinstance(attr, dict):
        if obj_id in attr:
            attr[obj_id] += n_counts
        else:
            attr[obj_id] = n_counts
    else:
        # If it's not a dict, reset it to a dict (legacy handling)
        setattr(scan, counter_name, {obj_id: n_counts})

    return getattr(scan, counter_name)[obj_id]


def get_counter(scan=None, counter_name=None):
    """
    Get the current count from a counter attribute.

    Parameters
    ----------
    scan : object
        The object to query.
    counter_name : str
        The name of the counter attribute.

    Returns
    -------
    int
        The current count value.
    """
    if scan is None:
        raise Exception("No object was passed!")
    if counter_name is None:
        raise Exception("No Name was passed!")

    _init_attr(scan=scan, attr_name=counter_name, attr_type=dict)
    obj_id = str(id(scan))

    attr = getattr(scan, counter_name)

    if isinstance(attr, dict):
        if obj_id in attr:
            return attr[obj_id]
        else:
            attr[obj_id] = 0
            return 0
    else:
        setattr(scan, counter_name, {obj_id: 0})
        return 0
