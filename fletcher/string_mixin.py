import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.arrays import ExtensionArray

try:
    # Only available in pandas 1.2+
    from pandas.core.strings.object_array import ObjectStringArrayMixin

    class _IntermediateExtensionArray(ExtensionArray, ObjectStringArrayMixin):
        pass


except ImportError:

    class _IntermediateExtensionArray(ExtensionArray):  # type: ignore
        pass


class StringSupportingExtensionArray(_IntermediateExtensionArray):
    def _str_contains(self, pat, case=True, flags=0, na=np.nan, regex=True):
        if not regex and case and hasattr(pc, "match_substring"):
            return type(self)(pc.match_substring(self.data, pat), dtype=pa.bool_())
        else:
            return super()._str_contains(pat, case, flags, na, regex)

    def _str_map(self, *args, **kwargs):
        return type(self)(super()._str_map(*args, **kwargs))

    def _str_startswith(self, pat, na=None):
        # TODO: This is currently not implemented in Arrow but only directly in the fr_strx accessor.
        return super()._str_startswith(pat, na)

    def _str_endswith(self, pat, na=None):
        # TODO: This is currently not implemented in Arrow but only directly in the fr_strx accessor.
        return super()._str_endswith(pat, na)
