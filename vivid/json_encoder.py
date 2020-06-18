"""json encoder allow nested object. clone from comet ml json_encoder.py"""

import json
import traceback
from datetime import date
from inspect import istraceback

import pandas as pd


def convert_pandas_dataframe_to_dict(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict()


def convert_pandas_series_to_array(value):
    if isinstance(value, pd.Series):
        return value.to_dict()


def convert_pandas_index_to_array(value: pd.Index):
    if isinstance(value, pd.Index):
        return value.tolist()


def convert_datetime_to_str(value):
    if isinstance(value, date):
        return str(value)


convert_functions = [
    convert_pandas_dataframe_to_dict,
    convert_pandas_series_to_array,
    convert_pandas_index_to_array,
    convert_datetime_to_str
]

try:
    import numpy


    def convert_numpy_array_pre_1_16(value):
        try:
            return numpy.asscalar(value)
        except (ValueError, IndexError, AttributeError, TypeError):
            return


    def convert_numpy_array_to_list(value):
        if isinstance(value, numpy.ndarray):
            return value.tolist()
        return


    def convert_numpy_array_post_1_16(value):
        try:
            return value.item()
        except (ValueError, IndexError, AttributeError, TypeError):
            return


    convert_functions.append(convert_numpy_array_post_1_16)
    convert_functions.append(convert_numpy_array_pre_1_16)
    convert_functions.append(convert_numpy_array_to_list)
except ImportError:
    pass


class NestedEncoder(json.JSONEncoder):
    """
    A JSON Encoder that converts floats/decimals to strings and allows nested objects
    """

    def default(self, obj):

        # First convert the object
        obj = self.convert(obj)

        # Check if the object is convertible
        try:
            json.JSONEncoder().encode(obj)
            return obj

        except TypeError:
            pass

        # Custom conversion
        if type(obj) == Exception or isinstance(obj, Exception) or type(obj) == type:
            return str(obj)

        elif istraceback(obj):
            return "".join(traceback.format_tb(obj)).strip()

        elif hasattr(obj, "repr_json"):
            return obj.repr_json()

        elif isinstance(obj, complex):
            return str(obj)

        else:
            try:
                return json.JSONEncoder.default(self, obj)

            except TypeError:
                return "%s not JSON serializable" % obj.__class__.__name__

    def floattostr(self, o, _inf=float("Inf"), _neginf=-float("-Inf"), nan_str="None"):
        if o != o:
            return nan_str

        else:
            return o.__repr__()

    def convert(self, obj):
        """
        Try converting the obj to something json-encodable
        """
        for converter in convert_functions:
            converted = converter(obj)

            if converted is not None:
                obj = converted
                break

        return obj
