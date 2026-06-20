"""Parser package for library-specific spectrum parsing."""
from .usgs_parser import parse_usgs_file as parse_usgs
from .relab_parser import parse_relab
from .rruff_parser import parse_rruff

PARSERS = {
    'usgs': parse_usgs,
    'relab': parse_relab,
    'rruff': parse_rruff,
}

def get_parser_for_path(path_str: str):
    """Return parser function based on path hints (RELAB/RRUFF/USGS).

    Args:
        path_str: File path string

    Returns:
        Parser function or None
    """
    lower = path_str.lower()
    if 'usgs' in lower:
        return parse_usgs
    if 'relab' in lower:
        return parse_relab
    if 'rruff' in lower:
        return parse_rruff
    return None
