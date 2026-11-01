import struct
import numpy as np
def pack_byte_strings(list_of_strings):
    """ Pack a list of byte strings into a single byte string

    Args:
        list_of_strings (List[str]): a list of byte strings

    Returns:
        str: a single byte string
    """
    # each string corresponds to a latent variable
    lengths = [len(s) for s in list_of_strings] # length of each string
    packed = b''.join(list_of_strings) # concatenate all strings
    # save the lengths of each string as uint32 'I'
    packed = struct.pack(f'{len(lengths)}I', *lengths) + packed
    # save the number of latent variables as a uint8 'B'
    packed = struct.pack(f'B', len(lengths)) + packed
    if False: # debug
        print(f'{len(packed)*8=} bits, {sum(lengths)*8=} bits')
        print(f'{lengths=}')
        decoded = unpack_byte_sting(packed)
        assert decoded == list_of_strings, f'{decoded=} should equal to {list_of_strings=}'
    return packed


def unpack_byte_string(string):
    """ Unpack a byte string into a list of byte strings.
    The input byte string should be packed by `pack_byte_strings()`.

    Args:
        string (str): a byte string packed by `pack_byte_strings()`

    Returns:
        List[str]: a list of byte strings
    """
    # read the number of latent variables
    _len = 1
    num, string = struct.unpack('B', string[:_len])[0], string[_len:]
    # read the lengths of each string
    _len = num * 4
    lengths, string = struct.unpack(f'{num}I', string[:_len]), string[_len:]
    assert sum(lengths) == len(string), f'{sum(lengths)=} should equal to {len(string)=}'
    # split the string into num strings
    edges = np.cumsum((0,) + lengths, dtype=np.uint32)
    strings_all = [string[edges[i]:edges[i+1]] for i in range(num)]
    return strings_all

