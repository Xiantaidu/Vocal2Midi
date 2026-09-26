def t0_nstep_to_ts(t0: float, nsteps: int) -> list:
    if nsteps <= 0:
        return [t0]
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]
