import io
from d3m.metadata import base

def metadata_to_str(metadata: base.Metadata, selector: base.Selector = None) -> str:
    buf = io.StringIO()
    metadata.pretty_print(selector, buf)
    return buf.getvalue()