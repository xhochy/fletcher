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

    def _str_isalnum(self):
        if hasattr(pc, "utf8_is_alnum"):
            return type(self)(pc.utf8_is_alnum(self.data))
        else:
            return super()._str_isalnum()

    def _str_isalpha(self):
        if hasattr(pc, "utf8_is_alpha"):
            return type(self)(pc.utf8_is_alpha(self.data))
        else:
            return super()._str_isalpha()

    def _str_isdecimal(self):
        if hasattr(pc, "utf8_is_decimal"):
            return type(self)(pc.utf8_is_decimal(self.data))
        else:
            return super()._str_isdecimal()

    def _str_isdigit(self):
        if hasattr(pc, "utf8_is_digit"):
            return type(self)(pc.utf8_is_digit(self.data))
        else:
            return super()._str_isdigit()

    def _str_islower(self):
        if hasattr(pc, "utf8_is_lower"):
            return type(self)(pc.utf8_is_lower(self.data))
        else:
            return super()._str_islower()

    def _str_isnumeric(self):
        if hasattr(pc, "utf8_is_numeric"):
            return type(self)(pc.utf8_is_numeric(self.data))
        else:
            return super()._str_isnumeric()

    def _str_isspace(self):
        if hasattr(pc, "utf8_is_space"):
            return type(self)(pc.utf8_is_space(self.data))
        else:
            return super()._str_isspace()

    def _str_istitle(self):
        if hasattr(pc, "utf8_is_title"):
            return type(self)(pc.utf8_is_title(self.data))
        else:
            return super()._str_istitle()

    def _str_isupper(self):
        if hasattr(pc, "utf8_is_upper"):
            return type(self)(pc.utf8_is_upper(self.data))
        else:
            return super()._str_isupper()
