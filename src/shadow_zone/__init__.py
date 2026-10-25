"""Shadow Zone Framing Model: called-strike probability, catcher and umpire effects, runs."""

try:  # use the OS trust store; some networks re-sign TLS with a private root CA
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

__version__ = "0.1.0"
