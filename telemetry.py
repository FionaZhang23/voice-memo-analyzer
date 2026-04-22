import os
from azure.monitor.opentelemetry import configure_azure_monitor

_TELEMETRY_INITIALIZED = False


def init_telemetry():
    """
    Initialize Azure Application Insights / Azure Monitor telemetry.
    This must be called before Flask or Azure SDK imports.
    """
    global _TELEMETRY_INITIALIZED

    if _TELEMETRY_INITIALIZED:
        return

    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING not set")

    configure_azure_monitor(connection_string=connection_string)
    _TELEMETRY_INITIALIZED = True
    print("Application Insights initialized.")