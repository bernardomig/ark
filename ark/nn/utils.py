
def round_channels(channels: int, divisor: int = 8):
    r"""Rounds a number to the nearest multiple of the divisor. 

    Useful to ensure that number of channels is GPU friendly 
    (multiple of 8 or 4), in networks scaled by a width multiplier.
    """
    c = int(channels + divisor / 2) // divisor * divisor
    c = c + divisor if c < (0.9 * channels) else c
    return c
