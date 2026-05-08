import nmrkit as nk


def _is_qf_indirect(data):
    return data.ndim > 1 and data.dimensions[1].domain_metadata.get("fnmode") == 1


def _default_indirect_size(data):
    current_size = data.dimensions[1].size
    if _is_qf_indirect(data):
        return max(1024, current_size)
    return 2048


def process(data, **kwargs):
    """Basic 2D NMR data processing workflow.

    Parameters
    ----------
    data : nmrkit.NMRData
        Input NMR data.
    **kwargs
        Additional processing parameters.

        zf_size_dim1 : int, optional
            Zero filling size for dimension 1.
        zf_size_dim2 : int, optional
            Zero filling size for dimension 2.
        window_indirect : str or bool, optional
            Indirect-dimension window. Defaults to "cosine" for QF data and
            disabled for phase-sensitive indirect dimensions.

    Returns
    -------
    data : nmrkit.NMRData
        Processed NMR data.
    """
    # Compensate JEOL Delta digital-filter delay in the time domain, before FT.
    if data.source_format == "delta":
        data = nk.compensate_digital_filter_delay(data)

    # Direct dimension processing
    data = nk.zf(data, dim=0, size=kwargs.get("zf_size_dim1"))
    data = nk.ft(data, dim=0)

    if data.source_format != "delta":
        data = nk.correct_digital_filter_phase(data)

    data = nk.autophase(data, dim=0)

    # Indirect dimension processing
    data = nk.complexify_indirect(
        data, mode=kwargs.get("indirect_complex_mode", "auto")
    )

    window_indirect = kwargs.get(
        "window_indirect", "cosine" if _is_qf_indirect(data) else None
    )
    if window_indirect in {True, "cosine"}:
        data = nk.cosine(data, dim=1)
    elif window_indirect == "cosine2":
        data = nk.cosine(data, dim=1, squared=True)
    elif window_indirect not in {False, None}:
        raise ValueError(f"Unknown indirect window: {window_indirect}")

    zf_size_dim2 = kwargs.get("zf_size_dim2")
    if zf_size_dim2 is None:
        zf_size_dim2 = _default_indirect_size(data)

    data = nk.zf(data, dim=1, size=zf_size_dim2)
    data = nk.ft(data, dim=1)

    if kwargs.get("autophase_indirect", not _is_qf_indirect(data)):
        data = nk.autophase(data, dim=1)

    # Baseline correction (opt-in, disabled by default)
    baseline = kwargs.get("baseline", False)
    if baseline:
        bc_kwargs = {}
        if isinstance(baseline, str):
            bc_kwargs["method"] = baseline
        data = nk.baseline_correct(data, **bc_kwargs)

    return data
