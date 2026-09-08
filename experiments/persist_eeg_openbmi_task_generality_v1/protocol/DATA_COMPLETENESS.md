# OpenBMI ERP / SSVEP data completeness

Status: `COMPLETE`.

The preprocessed local cache contains all 54 subjects and both cache sessions for ERP and SSVEP. Official NEMAR BIDS sidecars were read directly for event semantics, channel names, and original sampling rate. ERP has one logged one-trial cache exclusion; it is an objective cache-integrity exception, not a subject or outcome selection.

Missing cells: none.

The cache has no embedded channel-name sidecar; cache channel order is accepted only because every corresponding NEMAR BIDS train recording exposes the same 62-channel order and the cache schema matches that canonical count.
