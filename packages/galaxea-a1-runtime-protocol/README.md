# Galaxea A1 Runtime Protocol

`galaxea-a1-runtime-protocol` is the lightweight, hardware-free distribution
shared by the Galaxea A1 Runtime service and its clients. It contains only the
versioned Protobuf schema, generated stubs, scalar DTO/contracts, endpoint
validation, codecs, and thin gRPC client.

```bash
python -m pip install galaxea-a1-runtime-protocol
```

The distribution does not host a server, open ROS, acquire a command lease, or
own a watchdog. Those safety and lifecycle responsibilities remain inside the
`galaxea-a1-runtime` repository. It is developed and released from that same
repository; there is intentionally no separate protocol repository.
