import numpy as np
import pytest

from galaxea_a1_runtime.inference.msgpack_numpy import Packer


def test_lingbot_codec_rejects_pointer_bearing_object_arrays():
    packer = Packer()

    with pytest.raises(ValueError, match="unsupported inference array dtype"):
        packer.pack({"state": np.asarray([object()], dtype=object)})
